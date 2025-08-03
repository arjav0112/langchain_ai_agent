import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token (HF_TOKEN) not found in environment variables.")
    st.stop()

# Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
search = DuckDuckGoSearchRun(name="Search")

# Embed and store
@st.cache_resource(show_spinner="Indexing and embedding PDF...")
def load_doc_cached(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    splits = text_splitter.split_documents(docs)
    db = FAISS.from_documents(splits, HuggingFaceEmbeddings(      
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    return db.as_retriever()

st.title("🔎 LangChain - Chat with search")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Check if a new PDF is uploaded
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

    # Only load if not already in session_state
    if "retriever_pdf_hash" not in st.session_state or st.session_state.retriever_pdf_hash != pdf_hash:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        try:
            with st.spinner("Processing your PDF..."):
                st.session_state.retriever = load_doc_cached(temp_pdf_path)
                st.session_state.retriever_pdf_hash = pdf_hash
            st.success("PDF successfully processed!")
        except Exception as e:
            st.error(f"Failed to process the PDF: {e}")

retriever_tool = None
if "retriever" in st.session_state:

    retriever_tool = Tool.from_function(
        name="pdf-search",
        func=st.session_state.retriever.invoke,
        description="Search the uploaded PDF for relevant information"
    )

# Session state init
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Chat input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]
    if retriever_tool:
        tools.append(retriever_tool)

    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
