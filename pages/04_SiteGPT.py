from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from fake_useragent import UserAgent
import streamlit as st


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

# Scrapping .xml
CLOUDFARE_URL = "https://developers.cloudflare.com/sitemap-0.xml"

# AI Settings
st.session_state["choose_phase"] = False


class ChatCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, *args, **kwargs):
        if st.session_state["choose_phase"]:
            self.message = ""
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        if st.session_state["choose_phase"]:
            save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        if st.session_state["choose_phase"]:
            self.message += token
            self.message_box.markdown(self.message)


def set_api_key():
    st.session_state["api_key"] = st.session_state["api_key_input"]
    st.session_state["api_key_input"] = ""


def reset_api_key():
    del st.session_state["api_key"]


llm_settings = {
    "temperature": 0.1,
    "model": "gpt-4o-mini",
    "streaming": True,
    "callbacks": [
        ChatCallbackHandler(),
    ],
}

if "api_key" in st.session_state:
    with st.sidebar:
        st.text("Your key applied successfully")
        st.button("Reset", type="primary", on_click=reset_api_key)
else:
    st.sidebar.text_input(
        "Apply your api key",
        key="api_key_input",
        on_change=set_api_key,
    )

if "api_key" in st.session_state:
    llm_settings["api_key"] = st.session_state["api_key"]

llm = ChatOpenAI(**llm_settings)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answer that have the highest score (more helpful) and favor the most recent one.

            Cite score, sources and return the sources of the answers as it is, do not change it.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

ua = UserAgent()


# Functions
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


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    st.session_state["choose_phase"] = True
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()
    nav = soup.find("nav", class_="sidebar")
    if nav:
        nav.decompose()
    aside = soup.find("aside", class_="right-sidebar-container")
    if aside:
        aside.decompose()
    return str(soup.get_text()).replace("\\n", " ")


@st.cache_data(show_spinner="Loading website...")
def load_website():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        CLOUDFARE_URL,
        filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    loader.headers = {"User-Agent": ua.random}
    docs = loader.load_and_split(text_splitter=splitter)
    embedding_settings = {}
    if "api_key" in st.session_state:
        embedding_settings["api_key"] = st.session_state["api_key"]
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(**embedding_settings),
    )
    return vector_store.as_retriever()


# Streamlit
if "api_key" in st.session_state:
    retriever = load_website()
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    query = st.chat_input("Ask a question to the website.")
    if query:
        send_message(query, "human")
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            chain.invoke(query)
        st.session_state["choose_phase"] = False
else:
    st.markdown(
        """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by apply your api key on the sidebar.
    """
    )
    st.session_state["messages"] = []

# For challenge

with st.sidebar:
    st.link_button(
        "Go to repository", "https://github.com/sweetandsourkiss/fullstack-gpt"
    )
    st.code(
        '''
    from langchain.document_loaders import SitemapLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.callbacks.base import BaseCallbackHandler
    from fake_useragent import UserAgent
    import streamlit as st


    st.set_page_config(
        page_title="SiteGPT",
        page_icon="🖥️",
    )

    # Scrapping .xml
    CLOUDFARE_URL = "https://developers.cloudflare.com/sitemap-0.xml"

    # AI Settings
    st.session_state["choose_phase"] = False


    class ChatCallbackHandler(BaseCallbackHandler):

        def on_llm_start(self, *args, **kwargs):
            if st.session_state["choose_phase"]:
                self.message = ""
                self.message_box = st.empty()

        def on_llm_end(self, *args, **kwargs):
            if st.session_state["choose_phase"]:
                save_message(self.message, "ai")

        def on_llm_new_token(self, token, *args, **kwargs):
            if st.session_state["choose_phase"]:
                self.message += token
                self.message_box.markdown(self.message)


    def set_api_key():
        st.session_state["api_key"] = st.session_state["api_key_input"]
        st.session_state["api_key_input"] = ""


    def reset_api_key():
        del st.session_state["api_key"]


    llm_settings = {
        "temperature": 0.1,
        "model": "gpt-4o-mini",
        "streaming": True,
        "callbacks": [
            ChatCallbackHandler(),
        ],
    }

    if "api_key" in st.session_state:
        with st.sidebar:
            st.text("Your key applied successfully")
            st.button("Reset", type="primary", on_click=reset_api_key)
    else:
        st.sidebar.text_input(
            "Apply your api key",
            key="api_key_input",
            on_change=set_api_key,
        )

    if "api_key" in st.session_state:
        llm_settings["api_key"] = st.session_state["api_key"]

    llm = ChatOpenAI(**llm_settings)

    answers_prompt = ChatPromptTemplate.from_template(
        """
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
        Then, give a score to the answer between 0 and 5.

        If the answer answers the user question the score should be high, else it should be low.

        Make sure to always include the answer's score even if it's 0.

        Context: {context}
                                                    
        Examples:
                                                    
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0
                                                    
        Your turn!

        Question: {question}
        """
        )

        choose_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Use ONLY the following pre-existing answers to answer the user's question.

                    Use the answer that have the highest score (more helpful) and favor the most recent one.

                    Cite score, sources and return the sources of the answers as it is, do not change it.

                    Answers: {answers}
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        ua = UserAgent()


        # Functions
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


        def get_answers(inputs):
            docs = inputs["docs"]
            question = inputs["question"]
            answers_chain = answers_prompt | llm
            return {
                "question": question,
                "answers": [
                    {
                        "answer": answers_chain.invoke(
                            {"question": question, "context": doc.page_content}
                        ).content,
                        "source": doc.metadata["source"],
                        "date": doc.metadata["lastmod"],
                    }
                    for doc in docs
                ],
            }


        def choose_answer(inputs):
            st.session_state["choose_phase"] = True
            answers = inputs["answers"]
            question = inputs["question"]
            choose_chain = choose_prompt | llm
            condensed = "\n\n".join(
                f"Answer: {answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
                for answer in answers
            )
            return choose_chain.invoke(
                {
                    "question": question,
                    "answers": condensed,
                }
            )


        def parse_page(soup):
            header = soup.find("header")
            if header:
                header.decompose()
            nav = soup.find("nav", class_="sidebar")
            if nav:
                nav.decompose()
            aside = soup.find("aside", class_="right-sidebar-container")
            if aside:
                aside.decompose()
            return str(soup.get_text()).replace("\\n", " ")


        @st.cache_data(show_spinner="Loading website...")
        def load_website():
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200,
            )
            loader = SitemapLoader(
                CLOUDFARE_URL,
                filter_urls=[
                    r"^(.*\/workers-ai\/).*",
                    r"^(.*\/ai-gateway\/).*",
                    r"^(.*\/vectorize\/).*",
                ],
                parsing_function=parse_page,
            )
            loader.requests_per_second = 2
            loader.headers = {"User-Agent": ua.random}
            docs = loader.load_and_split(text_splitter=splitter)
            embedding_settings = {}
            if "api_key" in st.session_state:
                embedding_settings["api_key"] = st.session_state["api_key"]
            vector_store = FAISS.from_documents(
                docs,
                OpenAIEmbeddings(**embedding_settings),
            )
            return vector_store.as_retriever()


        # Streamlit
        if "api_key" in st.session_state:
            retriever = load_website()
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()
            query = st.chat_input("Ask a question to the website.")
            if query:
                send_message(query, "human")
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                with st.chat_message("ai"):
                    chain.invoke(query)
                st.session_state["choose_phase"] = False
        else:
            st.markdown(
                """
            # SiteGPT
                    
            Ask questions about the content of a website.
                    
            Start by apply your api key on the sidebar.
            """
            )
            st.session_state["messages"] = []
    '''
    )
