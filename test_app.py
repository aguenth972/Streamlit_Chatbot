import streamlit as st
import requests
import os
import time
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables. This is crucial for the embeddings model.
load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Chatbot Test App", layout="wide")
st.title("Land Ordinance Chatbot - Test Interface")

# --- Constants ---
CHROMA_PATH = "chroma_db"
TEMP_DIR = "temp_files"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- Helper Functions for Vector Store Creation ---

def create_vector_store(file_path):
    """Orchestrates the creation of the vector store."""
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file type. Please upload a .pdf or .txt file.")
        return

    with st.status("Processing document...", expanded=True) as status:
        st.write("Loading document...")
        documents = loader.load()

        st.write("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        st.write("Initializing embedding model...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )

        st.write(f"Creating and saving vector store to '{CHROMA_PATH}'...")
        if os.path.exists(CHROMA_PATH):
            st.warning(f"Existing database found at '{CHROMA_PATH}'. It will be overwritten.")
        
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        status.update(label="Vector store created successfully!", state="complete", expanded=False)

    st.success(f"Vector database created from '{os.path.basename(file_path)}' and saved in '{CHROMA_PATH}'.")

# --- Streamlit App UI ---

tab1, tab2 = st.tabs(["1. Create Vector Store", "2. Chat with API"])

# --- TAB 1: VECTOR STORE CREATION ---
with tab1:
    st.header("Create or Overwrite the Vector Store")
    st.info("Upload a document (PDF or TXT) to build the ChromaDB vector store. This will be used by the API.")

    uploaded_file = st.file_uploader("Choose a document...", type=["pdf", "txt"])

    if st.button("Create Vector Store"):
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run the vector store creation process
            create_vector_store(temp_file_path)

            # Clean up the temporary file
            os.remove(temp_file_path)
        else:
            st.warning("Please upload a file first.")

    st.subheader("Current Database Status")
    if os.path.exists(CHROMA_PATH):
        st.success(f"A database currently exists at: `{CHROMA_PATH}`")
    else:
        st.warning("No database found. Please create one.")

# --- TAB 2: CHAT INTERFACE ---
with tab2:
    st.header("Chat with Your Ordinance Expert")
    st.info("This interface sends requests to the running FastAPI server (`api.py`). Make sure it is running!")

    # Input for the API URL
    api_url = st.text_input("API URL", "http://127.0.0.1:8000/chat")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the ordinance..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Prepare the request payload
                # Convert session state format to the List[Tuple[str, str]] the API expects
                chat_history_for_api = [
                    (msg["content"]) for msg in st.session_state.messages if msg["role"] == 'user' or msg["role"] == 'assistant'
                ]
                # The API expects pairs, so we need to process this. A simpler way for this test app:
                # We can just send the last few turns or structure it properly. For simplicity, let's match the required format.
                history_tuples = []
                temp_history = st.session_state.messages[:-1] # Exclude the current question
                for i in range(0, len(temp_history), 2):
                    if i+1 < len(temp_history):
                        history_tuples.append( (temp_history[i]['content'], temp_history[i+1]['content']) )


                payload = {"question": prompt, "chat_history": history_tuples}
                
                with st.spinner("Thinking..."):
                    response = requests.post(api_url, json=payload)
                    response.raise_for_status() # Raise an exception for bad status codes
                    full_response = response.json().get("answer", "No answer found.")

                message_placeholder.markdown(full_response)

            except requests.exceptions.RequestException as e:
                full_response = f"**Error:** Could not connect to the API at `{api_url}`. **Is the `api.py` server running?**\n\nDetails: {e}"
                message_placeholder.error(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})