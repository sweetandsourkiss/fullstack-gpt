import streamlit as st
from openai import OpenAI
from pathlib import Path
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.retrievers import WikipediaRetriever
from typing_extensions import override
from openai import AssistantEventHandler

st.title("OpenAI Assistant")


def make_file(keyword, content):
    # 저장할 경로 설정
    save_text_path = Path("./.cache/research_files")
    save_text_path.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성 (존재하지 않으면)

    # 파일 저장 경로 설정
    file_path = "./.cache/research_files/" + keyword + ".txt"

    # 파일 저장 (텍스트 모드)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == "thread.run.requires_action":
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.message = ""
            self.message_box = st.empty()
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "research_on_ddg":
                tool_outputs.append({"tool_call_id": tool.id, "output": "ddg"})
            elif tool.function.name == "research_on_wp":
                tool_outputs.append({"tool_call_id": tool.id, "output": "wp"})

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                self.message += text
                self.message_box.markdown(self.message)
                print(text, end="", flush=True)


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
    )
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
        for annotation in message.content[0].text.annotations:
            print(f"Source: {annotation.file_citation}")


def set_api_key():
    st.session_state["api_key"] = st.session_state["api_key_input"]
    st.session_state["api_key_input"] = ""


def reset_api_key():
    del st.session_state["api_key"]


if "api_key" in st.session_state:
    with st.sidebar:
        st.text("Your key applied successfully")
        st.button("Reset", type="primary", on_click=reset_api_key)

    if "client" in st.session_state:
        client = st.session_state["client"]
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]
    else:
        client = OpenAI(api_key=st.session_state["api_key"])
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="You help research with two tools:DuckDuckGo and Wikipedia. Return the result with tool name on the top.",
            model="gpt-4o-mini",
            temperature=0.1,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "research_on_ddg",
                        "description": "Get the research result for the query on the DuckDuckGo.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "full string of user's input",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "research_on_wp",
                        "description": "Get the research result for the query on the Wikipedia.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "full string of user's input",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
            ],
        )
        thread = client.beta.threads.create()
        st.session_state["client"] = client
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if messages:
        messages = list(messages)
        messages.reverse()
        for message in messages:
            st.chat_message(message.role).write(message.content[0].text.value)

    question = st.chat_input("Ask to chatbot anything!!")
    if question:
        with st.chat_message("user"):
            st.write(question)
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question,
        )

        with st.chat_message("ai"):
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()


else:
    st.sidebar.text_input(
        "Apply your api key",
        key="api_key_input",
        on_change=set_api_key,
    )
