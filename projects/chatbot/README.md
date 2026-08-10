# Chatbot — HuzaifaBot (Streamlit + LangGraph)

A stateful personal AI assistant built with **LangGraph**, **Streamlit**, and **Groq** (`llama-3.1-8b-instant`). It talks in a natural, casual tone (English mixed with Roman Urdu) and remembers context across turns using a checkpointer.

## ✨ Features

- Multi-turn conversation memory via LangGraph checkpointing (`thread_id`).
- System prompt that personalizes the assistant (HuzaifaBot) for its owner.
- Simple chat UI built with Streamlit.
- Fast inference on Groq's free tier.

## 📂 Project Structure

```
chatbot/
├── config.py    # ChatGroq model setup (reads groq_api_key from .env)
├── main.py      # LangGraph workflow: state, node, graph compile + checkpointer
├── frontend.py  # Streamlit UI that calls the workflow
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

- `config.py` loads the Groq API key from `.env` and creates the `ChatGroq` model.
- `main.py` defines a single-node `StateGraph`. State holds the message list with LangGraph's `add_messages` reducer. The graph is compiled with a `MemorySaver` checkpointer so each `thread_id` keeps its own conversation memory.
- `frontend.py` renders the chat UI, calls `workflow.invoke(...)` for each user message, and appends replies to the session.

## ⚠️ Deployment Notes

- `MemorySaver` stores chat history only in RAM — it is lost when the server restarts or sleeps (e.g., free-tier hosts like Streamlit Community Cloud).
- The fixed `thread_id: "1"` is shared by all users, so conversations are not isolated per user.
- For production, replace `MemorySaver` with a persistent checkpointer (e.g., `PostgresSaver` from `langgraph-checkpoint-postgres`) and use a unique `thread_id` per user session.

## 🔑 Key Takeaways

1. Compile the graph with a checkpointer to make it **stateful**: `graph.compile(checkpointer=checkpoint)`.
2. Same `thread_id` = same memory; different `thread_id` = isolated conversations.
3. Streamlit is a quick way to expose a LangGraph workflow as a chat app.
