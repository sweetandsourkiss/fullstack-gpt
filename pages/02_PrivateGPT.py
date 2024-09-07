import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📄",
)


class ChatCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def set_api_key():
    st.session_state["api_key"] = st.session_state["input"]
    st.session_state["input"] = ""
    st.toast("Api key applied")


def reset_api_key():
    del st.session_state["api_key"]
    st.toast("Api key reset")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 파일 내용을 읽음
    file_content = file.read()

    # 저장할 경로 설정
    cache_dir_path = Path("./.cache/private_files")
    cache_dir_path.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성 (존재하지 않으면)

    # 파일 저장 경로 설정
    file_path = cache_dir_path / file.name

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Embeddings 저장할 경로 생성
    embeddings_dir_path = Path(f"./.cache/private_embeddings/{file.name}")
    embeddings_dir_path.mkdir(
        parents=True, exist_ok=True
    )  # 디렉토리 생성 (존재하지 않으면)

    # LocalFileStore 설정
    cache_dir = LocalFileStore(str(embeddings_dir_path))

    # 파일을 분할하는 Text Splitter 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # 파일 로드 및 분할
    loader = UnstructuredFileLoader(str(file_path))
    docs = loader.load_and_split(text_splitter=splitter)

    # Embeddings 생성 및 캐시 처리
    embedding_settings = {
        "model": "mistral:latest",
    }

    if "api_key" in st.session_state:
        embedding_settings["api_key"] = st.session_state["api_key"]

    embeddings = OllamaEmbeddings(**embedding_settings)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # FAISS를 사용한 VectorStore 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    st.text_input("your api key", key="input", on_change=set_api_key)

llm_settings = {
    "model": "mistral:latest",
    "temperature": 0.1,
    "streaming": True,
    "callbacks": [
        ChatCallbackHandler(),
    ],
}

if "api_key" in st.session_state:
    llm_settings["api_key"] = st.session_state["api_key"]
    with st.sidebar:
        st.write(f"your key: {st.session_state['api_key']}")
        st.button("Reset", type="primary", on_click=reset_api_key)

llm = ChatOllama(**llm_settings)


prompt = ChatPromptTemplate.from_template(
    """
        Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
     
        Context: {context}
        Qustion: {question}
        """
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []
