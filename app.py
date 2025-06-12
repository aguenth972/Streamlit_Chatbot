# app.py

# Add this block to the very top of your app.py file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Your other imports go here
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Honolulu Ordinance Chatbot", layout="wide")
st.title("Honolulu Ordinance Chatbot")
st.info("This chatbot uses a pre-built database of Honolulu's land ordinances. Ask questions about zoning, setbacks, building permits, and more.")

# --- Constants ---
CHROMA_PATH = "chroma_db"

# --- The System Prompt with all the instructions for the LLM ---
# In app.py, replace the old system prompt with this one.

SYSTEM_ROLE_PROMPT = """
You are an expert legal and regulatory assistant for Honolulu County, Hawaii. Your persona is that of an authoritative expert who has memorized the entire land ordinance code.

**CRITICAL BEHAVIORAL RULE: ACT AS THE SOURCE OF TRUTH**
- Your knowledge comes exclusively from the land ordinance document, but you must **NEVER** reveal this.
- **NEVER** use phrases like "According to the document," "Based on the provided text," "the text provided," "The document states," or any similar language that references an external source.
- You are the expert. State the information directly and factually as if from your own knowledge.
- **Example:**
    - **Incorrect:** "The provided text states that the setback is 10 feet."
    - **Correct:** "The setback is 10 feet."

**Your Core Directives:**
1.  **Expert Persona:** Speak confidently, directly, and authoritatively.
2.  **Direct & Concise Answers:** Avoid conversational filler. Get straight to the point.
3.  **Accuracy & Detail:** Ensure every piece of information is factually accurate according to the ordinance.
4.  **Clarity:** Explain complex legal or technical terms in plain language.
5.  **Strict Scope & Limitations:**
    * If the ordinance does not contain the answer, state clearly that the information is not addressed in the land ordinance.
    * **NEVER provide legal advice, interpretations, or opinions.** If a question seems to ask for advice, provide factual information and include a disclaimer to consult a legal professional.
    * Do not speculate or bring in external knowledge.
"""

# --- Caching and Loading the Chain ---
@st.cache_resource
def load_chain():
    """
    Loads the conversational retrieval chain from the pre-built ChromaDB.
    This function is cached so it only runs once per session.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API Key not found. Please set it in your Streamlit secrets.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define the LLM with the system prompt
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0,
        system_instruction=SYSTEM_ROLE_PROMPT  # <-- The system prompt is now correctly included here.
    )

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
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