import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
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

def build_vector_store_in_memory(text_chunks):
    """Generate embeddings and save to st.session_state to avoid multi-user data leakage."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    batch_size = 5  # Process 5 chunks at a time to handle rate limits
    
    progress_text = "Embedding chunks safely into your session memory..."
    my_bar = st.progress(0, text=progress_text)
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size

    # Reset or initialize the session-specific vector store
    st.session_state.vector_store = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            st.session_state.vector_store.add_texts(batch)
            
        current_batch = (i // batch_size) + 1
        my_bar.progress(current_batch / total_batches, text=progress_text)
        
        # Sleep to let the free-tier API rate limits reset
        if current_batch < total_batches:
            time.sleep(10) 
            
    my_bar.empty()

def get_conversational_chain():
    """Set up the LLM chain with a flexible prompt template."""
    prompt_template = """
    You are a highly intelligent and helpful assistant. 
    First, look at the provided Context to see if it helps answer the user's Question. 
    If the Context contains relevant information, use it to build your answer.
    If the Context does NOT contain the answer, do not apologize or say it is unavailable. Instead, answer the question using your own general knowledge to the best of your ability.

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
    """Search the session-isolated FAISS instance and generate an answer."""
    try:
        # Pull the vector store directly from the current user's session state
        vector_store = st.session_state.vector_store
        
        # Similarity search for top 4 documents
        docs = vector_store.similarity_search(user_question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        chain = get_conversational_chain()
        response = chain.invoke({"context": context, "question": user_question})
        return response
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "⏳ **Rate Limit Hit:** Google's free API is cooling down. Please wait about 60 seconds and ask your question again!"
        else:
            return f"⚠️ **An unexpected error occurred:** {error_msg}"

def main():
    st.header("DocuChat AI 📄🤖")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize the vector store key in session state if not present
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

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
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Build and store directly in the user's isolated session state
                    build_vector_store_in_memory(text_chunks)
                    st.success("Processing Complete! Your session index is ready.")

    # Accept user input via chat interface
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Guard: Check session state instead of the local disk directory
        if st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF from the sidebar first.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_question(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()