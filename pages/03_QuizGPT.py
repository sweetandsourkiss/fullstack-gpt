import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json

# streamlit config
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


# class
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


# functions
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(
    show_spinner="Making quiz...",
)
def run_quiz_chain(level):
    prompt = PromptTemplate.from_template(
        f"Make a {level}-level quiz about {st.session_state['key_word']}"
    )
    chain = prompt | llm
    response = chain.invoke({})
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(term)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def set_api_key():
    st.session_state["api_key"] = st.session_state["api_key_input"]
    st.session_state["api_key_input"] = ""
    st.toast("Api key applied")


def reset_api_key():
    del st.session_state["api_key"]
    st.toast("Api key reset")


def set_key_word():
    st.session_state["key_word"] = st.session_state["key_word_input"]
    st.session_state["api_key_input"] = ""


def go_back():
    del st.session_state["key_word"]
    run_quiz_chain.clear()
    st.session_state["level"] = "EASY"


def set_level():
    st.session_state["level"] = st.session_state["level_input"]


# chain settings

output_parser = JsonOutputParser()

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm_settings = {
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "streaming": True,
    "callbacks": [
        StreamingStdOutCallbackHandler(),
    ],
}

if "api_key" in st.session_state:
    llm_settings["api_key"] = st.session_state["api_key"]
    with st.sidebar:
        encoded_key = st.session_state["api_key"][:8] + "..."
        st.write(f"your key: {encoded_key}")
        st.button("Reset", type="primary", on_click=reset_api_key)
else:
    with st.sidebar:
        st.text_input(
            "your api key",
            key="api_key_input",
            on_change=set_api_key,
        )


llm = ChatOpenAI(**llm_settings).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

# streamlit
st.title("QuizGPT")

if "key_word" in st.session_state:
    response = run_quiz_chain(st.session_state["level"])
    st.button("Reset Test", type="primary", on_click=go_back)
    st.subheader(
        f"Quiz about \"{st.session_state['key_word']}\" ({st.session_state['level']})"
    )
    if response:
        awesome = True
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    awesome = False
                    st.error("Wrong")
                else:
                    awesome = False
            button = st.form_submit_button()
        if awesome:
            st.balloons()


else:
    st.selectbox(
        "Level of Quiz", ["EASY", "HARD"], key="level_input", on_change=set_level
    )
    st.text_input(
        "Type a key word for quiz.",
        key="key_word_input",
        on_change=set_key_word,
    )
    if "level" not in st.session_state:
        st.session_state["level"] = "EASY"

with st.sidebar:
    st.link_button(
        "Go to repository", "https://github.com/sweetandsourkiss/fullstack-gpt"
    )
    st.code(
        """
    import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json

# streamlit config
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


# class
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


# functions
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(
    show_spinner="Making quiz...",
)
def run_quiz_chain(level):
    prompt = PromptTemplate.from_template(
        f"Make a {level}-level quiz about {st.session_state['key_word']}"
    )
    chain = prompt | llm
    response = chain.invoke({})
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(term)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def set_api_key():
    st.session_state["api_key"] = st.session_state["api_key_input"]
    st.session_state["api_key_input"] = ""
    st.toast("Api key applied")


def reset_api_key():
    del st.session_state["api_key"]
    st.toast("Api key reset")


def set_key_word():
    st.session_state["key_word"] = st.session_state["key_word_input"]
    st.session_state["api_key_input"] = ""


def go_back():
    del st.session_state["key_word"]
    run_quiz_chain.clear()
    st.session_state["level"] = "EASY"


def set_level():
    st.session_state["level"] = st.session_state["level_input"]


# chain settings

output_parser = JsonOutputParser()

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm_settings = {
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "streaming": True,
    "callbacks": [
        StreamingStdOutCallbackHandler(),
    ],
}

if "api_key" in st.session_state:
    llm_settings["api_key"] = st.session_state["api_key"]
    with st.sidebar:
        encoded_key = st.session_state["api_key"][:8] + "..."
        st.write(f"your key: {encoded_key}")
        st.button("Reset", type="primary", on_click=reset_api_key)
else:
    with st.sidebar:
        st.text_input(
            "your api key",
            key="api_key_input",
            on_change=set_api_key,
        )


llm = ChatOpenAI(**llm_settings).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

# streamlit
st.title("QuizGPT")

if "key_word" in st.session_state:
    response = run_quiz_chain(st.session_state["level"])
    st.button("Reset Test", type="primary", on_click=go_back)
    st.subheader(
        f"Quiz about \"{st.session_state['key_word']}\" ({st.session_state['level']})"
    )
    if response:
        awesome = True
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    awesome = False
                    st.error("Wrong")
                else:
                    awesome = False
            button = st.form_submit_button()
        if awesome:
            st.balloons()


else:
    st.selectbox(
        "Level of Quiz", ["EASY", "HARD"], key="level_input", on_change=set_level
    )
    st.text_input(
        "Type a key word for quiz.",
        key="key_word_input",
        on_change=set_key_word,
    )
    if "level" not in st.session_state:
        st.session_state["level"] = "EASY"

"""
    )
