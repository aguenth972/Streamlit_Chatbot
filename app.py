# app.py

# --- Imports ---
import os
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from prompt import SYSTEM_ROLE_PROMPT
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Honolulu Ordinance Chatbot", layout="wide")
st.title("Honolulu Ordinance Chatbot")
st.info("This chatbot uses a pre-built database of Honolulu's land ordinances. Ask questions about zoning, setbacks, building permits, and more.")

# --- Constants ---
CHROMA_PATH = "chroma_db"

# --- Caching and Loading the Chain ---
@st.cache_resource
def load_chain():
    """
    Loads the conversational retrieval chain from the pre-built ChromaDB.
    This version uses an explicit ChatPromptTemplate for more reliable persona control.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API Key not found. Please set it in your Streamlit secrets.")
        st.stop()

    # Create the new, explicit prompt template
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_ROLE_PROMPT),
        HumanMessagePromptTemplate.from_template("Based on the following context, please answer the question:\n\nContext:\n{context}\n\nQuestion: {question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Initialize the components
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define the LLM (we no longer need the system_instruction parameter here)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0,
    )

    # Create the conversational chain, passing in our custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': qa_prompt},  # <-- This is the key change
        return_source_documents=False,
        return_generated_question=False
    )
    return qa_chain

# --- Main App Logic ---
if not os.path.exists(CHROMA_PATH):
    st.error(f"Database not found at '{CHROMA_PATH}'. Please ensure the `chroma_db` directory is in the GitHub repository.")
    st.stop()

try:
    qa = load_chain()
except Exception as e:
    st.error(f"Failed to load the chatbot model. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about building regulations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):
            history_tuples = []
            temp_history = st.session_state.messages[:-1]
            for i in range(0, len(temp_history), 2):
                if i + 1 < len(temp_history):
                    history_tuples.append((temp_history[i]['content'], temp_history[i+1]['content']))

            result = qa({"question": prompt, "chat_history": history_tuples})
            full_response = result.get("answer", "Sorry, I encountered an error.")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})