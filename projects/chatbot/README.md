# Chatbot — HuzaifaBot (Streamlit + LangGraph)

A stateful personal AI assistant built with **LangGraph**, **Streamlit**, and **Groq** (`llama-3.1-8b-instant`). It talks in a natural, casual tone (English mixed with Roman Urdu) and remembers context across turns using a checkpointer.

## ✨ Features

- Multi-turn conversation memory via LangGraph checkpointing.
- **Multiple independent conversations** — every new chat gets a fresh `thread_id` (UUID), and you can switch between past conversations from the sidebar.
- **Token streaming** — responses stream into the chat UI as they are generated (`stream_mode="messages"`).
- **History reload** — selecting an old conversation restores its full message history from the checkpointer.
- Simple chat UI built with Streamlit.
- Fast inference on Groq's free tier.

## 📂 Project Structure

```
chatbot/
├── config.py    # ChatGroq model setup (reads groq_api_key from .env)
├── main.py      # LangGraph workflow: state, node, graph compile + checkpointer
├── frontend.py  # Streamlit UI: streaming chat + multi-conversation management
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)

### Installation

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install streamlit langgraph langchain-groq python-dotenv

# 3. Create .env with your Groq key
groq_api_key=your_key_here
```

### Running

```bash
streamlit run frontend.py
```

## 🧠 How It Works

### `config.py`
Loads the Groq API key from `.env` and creates the `ChatGroq` model (`llama-3.1-8b-instant`).

### `main.py` — the LangGraph workflow
- Defines a **single-node** `StateGraph`.
- State holds the message list with LangGraph's `add_messages` reducer, which automatically appends messages and keeps the full conversation:

```python
class chatbot_state(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
```

- `chat_node` passes the accumulated messages to the model and returns the reply, which `add_messages` appends to the state.
- The graph is compiled with a `MemorySaver` checkpointer:

```python
checkpoint = MemorySaver()
workflow = graph.compile(checkpointer=checkpoint)
```

### `frontend.py` — the Streamlit UI
- Generates a unique `thread_id` (UUID) for every chat session:

```python
def generating_thread_id():
    return uuid.uuid4()
```

- Tracks open conversations in `st.session_state["chat_thread"]`.
- **`+New chat`** (`reset()`) creates a fresh thread and clears the UI.
- Selecting a sidebar conversation (`get_state`) loads its history back from the checkpointer and renders it.
- Sends user messages through the graph and **streams** the assistant reply:

```python
res = st.write_stream(
    message_chunks.content for message_chunks, metadata in workflow.stream(
        {"message": [HumanMessage(content=msg_input)]},
        config=config,
        stream_mode="messages"
    )
)
```

- A `delete_thread()` helper is included (currently commented out in the UI) that removes a thread from the sidebar and calls `workflow.checkpointer.delete_thread(thread_id)`.

## ⚠️ Deployment Notes

- `MemorySaver` stores chat history only in RAM — it is lost when the server restarts or sleeps (e.g., free-tier hosts like Streamlit Community Cloud).
- Conversations are now isolated per session via unique `thread_id`s, but history still does not survive a server restart.
- For production, replace `MemorySaver` with a persistent checkpointer (e.g., `PostgresSaver` from `langgraph-checkpoint-postgres`).

## 🔑 Key Takeaways

1. Compile the graph with a checkpointer to make it **stateful**: `graph.compile(checkpointer=checkpoint)`.
2. Same `thread_id` = same memory; different `thread_id` = isolated conversations — use UUIDs to separate users/sessions.
3. `workflow.stream(..., stream_mode="messages")` enables real-time token streaming in the UI.
4. `workflow.get_state()` reads a thread's history back; `workflow.checkpointer.delete_thread()` removes it.