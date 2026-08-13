import streamlit as st
from langchain_core.messages import HumanMessage
from main import workflow
import uuid

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

def generating_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset():
    thread_id = generating_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)

def get_state(thread_id):
    return workflow.get_state(config={"configurable":{"thread_id":thread_id}}).values.get("message", [])

def delete_thread(thread_id):
    st.session_state["chat_thread"].remove(thread_id)
    workflow.checkpointer.delete_thread(thread_id)
    if st.session_state["thread_id"] == thread_id:
        st.session_state["thread_id"] = generating_thread_id()
        st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generating_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = []
    st.session_state["chat_thread"].append(st.session_state["thread_id"])

st.sidebar.title("Chatbot")
st.sidebar.header("Conversation")
if st.sidebar.button("+New chat"):
    reset()
for thread_id in st.session_state["chat_thread"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = get_state(thread_id)
        temp_msgs = []
        for msg in messages:
            if isinstance(msg , HumanMessage):
                role= "user"
            else:
                role="assistant"
            temp_msgs.append({"role":role ,"content":msg.content})

        st.session_state["message_history"] = temp_msgs
    # if col2.button("X", key=f"delete_{thread_id}", use_container_width=True):
    #     delete_thread(thread_id)

    

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

msg_input = st.chat_input("Type your message here...")
if msg_input:
    st.session_state["message_history"].append({"role": "user", "content": msg_input})
    with st.chat_message("user"):
        st.text(msg_input)

    config = {"configurable":{"thread_id":st.session_state["thread_id"]}}
    with st.chat_message("assistant"):
        res = st.write_stream(
            message_chunks.content for message_chunks, metadata in workflow.stream(
                {"message": [HumanMessage(content=msg_input)]},
                config=config,
                stream_mode="messages"
            )
        )
        st.session_state["message_history"].append({"role": "assistant", "content": res})