import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# It's good practice to load .env for local testing, but we'll use Streamlit's secrets for deployment.
load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Honolulu Ordinance Chatbot", layout="wide")
st.title("Honolulu Ordinance Chatbot")
st.info("This chatbot uses a pre-built database of Honolulu's land ordinances. Ask questions about zoning, setbacks, building permits, and more.")

# --- Constants ---
CHROMA_PATH = "chroma_db"

# --- Caching and Loading the Chain ---
# Use Streamlit's caching to load the model and chain only once.
@st.cache_resource
def load_chain():
    """
    Loads the conversational retrieval chain from the pre-built ChromaDB.
    """
    # Check for the API key, trying Streamlit's secrets first
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API Key not found. Please set it in your Streamlit secrets.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    # Load the persistent vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0,
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        return_generated_question=False
    )
    return qa_chain

# --- Main App Logic ---
# Check if the database directory exists before loading
if not os.path.exists(CHROMA_PATH):
    st.error(f"Database not found at '{CHROMA_PATH}'. Please ensure the `chroma_db` directory is in the GitHub repository.")
    st.stop()

try:
    qa = load_chain()
except Exception as e:
    st.error(f"Failed to load the chatbot model. Error: {e}")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about building regulations..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            # Format chat history for the chain
            history_tuples = []
            temp_history = st.session_state.messages[:-1]
            for i in range(0, len(temp_history), 2):
                if i + 1 < len(temp_history):
                    history_tuples.append((temp_history[i]['content'], temp_history[i+1]['content']))
            
            # Get the answer from the chain
            result = qa({"question": prompt, "chat_history": history_tuples})
            full_response = result.get("answer", "Sorry, I encountered an error.")

        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})