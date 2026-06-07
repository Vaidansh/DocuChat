import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    """Split text into overlapping chunks based on user specifications."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate embeddings and save to a local FAISS index with rate limit handling."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    vector_store = None
    batch_size = 5  # Process 5 chunks at a time
    
    # Progress bar for the UI
    progress_text = "Embedding chunks... Please wait to avoid rate limits."
    my_bar = st.progress(0, text=progress_text)
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            vector_store.add_texts(batch)
            
        # Update progress bar
        current_batch = (i // batch_size) + 1
        my_bar.progress(current_batch / total_batches, text=progress_text)
        
        # Sleep for 10 seconds between batches to let the API rate limits reset
        if current_batch < total_batches:
            time.sleep(10) 
            
    vector_store.save_local("faiss_index")
    my_bar.empty() # Clear the progress bar when done

def get_conversational_chain():
    """Set up the LLM chain with a strict prompt template."""
    prompt_template = """
    You are a highly intelligent and helpful assistant. 
    First, look at the provided Context to see if it helps answer the user's Question. 
    If the Context contains relevant information, use it to build your answer.
    If the Context does NOT contain the answer, do not apologize or say it is unavailable. Instead, answer the question using your own general knowledge to the best of your ability.
    (Optional: You can subtly mention if you had to rely on general knowledge instead of the document, but always provide a helpful answer).

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
    chain = prompt | llm | StrOutputParser()
    return chain

def handle_user_question(user_question):
    """Embed question, search FAISS, and generate an answer with error handling."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    # Load the local FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    try:
        # Similarity search for top 4 documents
        docs = new_db.similarity_search(user_question, k=4)
        
        # Join chunk text as context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        chain = get_conversational_chain()
        response = chain.invoke({"context": context, "question": user_question})
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "⏳ **Rate Limit Hit:** Google's free API is cooling down after processing your documents. Please wait about 60 seconds and ask your question again!"
        else:
            return f"⚠️ **An unexpected error occurred:** {error_msg}"

def main():
    st.header("DocuChat AI 📄🤖")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for PDF processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click Process", 
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    # 1. Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    # 2. Split text
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Create Vector Store
                    get_vector_store(text_chunks)
                    st.success("Processing Complete! FAISS index saved.")

    # Accept user input via chat interface
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Guard: Check if FAISS index exists before answering
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process a PDF from the sidebar first.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_question(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()