# 07_Sqlite+Memory — Persistent Memory with SQLite Checkpointing

This folder shows how to give a LangGraph chatbot **real, persistent memory** using `SqliteSaver` — the checkpointer that stores conversation state in a SQLite database on disk, so it survives server restarts.

## 📖 Topic Explained in Detail

### Why do we need memory in LangGraph?

In earlier examples (`06_Persistence`) we used `InMemorySaver`, which keeps checkpoints in **RAM**. It works perfectly for learning, but it has one fatal flaw:

> Everything is lost the moment the Python process stops.

In production, an agent must remember conversations across:
- Server restarts and crashes
- Deployments / redeploys
- Time (hours, days, weeks)

For that, the checkpointer must write state to a **persistent storage layer**.

### What is SQLite?

**SQLite** is a lightweight, file-based relational database. Unlike PostgreSQL/MySQL, it needs **no separate server** — the whole database is a single `.db` file (here, `chatbot.db`) that lives on your machine. It is:
- Zero-configuration (no setup, no admin, no ports)
- Fast for small/medium workloads
- Widely supported and bundled with Python's standard library (`sqlite3`)

This makes it the ideal "next step" after `InMemorySaver`: persistence with almost no extra infrastructure.

### Why use SQLite in LangGraph? (vs. other checkpointers)

| Checkpointer | Storage | Server needed? | Survives restarts? | Best for |
|---|---|---|---|---|
| `InMemorySaver` | RAM | No | ❌ No | Tutorials, quick experiments |
| **`SqliteSaver`** | SQLite `.db` file | **No** | ✅ Yes | Local apps, lightweight production, edge cases |
| `PostgresSaver` | PostgreSQL | Yes | ✅ Yes | Large-scale production, multi-user, multi-process |

**Key reasons to pick SQLite:**
1. **Persistence across restarts** — state is written to disk after every node, so a restart doesn't wipe conversation memory.
2. **Zero setup** — no database server to install or configure; perfect for desktop apps, prototypes, and small bots.
3. **Cheap** — no cloud database costs.
4. **Familiar SQL** — you can inspect `chatbot.db` with any SQLite viewer.
5. **Thread isolation preserved** — the same `thread_id` mechanism keeps conversations separate, just stored more durably.

### What exactly is stored?

Each checkpoint (one after every node execution) is stored in SQLite tables. LangGraph creates internal tables (e.g. `checkpoints`, `checkpoint_writes`) inside `chatbot.db` that hold:
- The thread metadata (`thread_id`, `checkpoint_id`)
- The full graph state (all the messages) at that point
- The pending writes for each node

### The API you need to know

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

con = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=con)
workflow = graph.compile(checkpointer=checkpoint)
```

- `sqlite3.connect("chatbot.db")` opens (or creates) the database file.
- `check_same_thread=False` allows the connection to be shared across threads — required because LangGraph may call nodes from different threads (especially with async/streaming).
- `SqliteSaver(conn=con)` wraps the connection as a LangGraph checkpointer.
- Everything else is identical to `InMemorySaver` usage.

> **Important:** as of recent LangGraph versions, the import path is `langgraph.checkpoint.sqlite.SqliteSaver`. (Older docs used `langgraph.checkpoint.sqlite.aio` / factory helpers — use the direct class shown here.)

## 🧠 Code Concepts Explained in Detail

### 1. Chat state with `add_messages`

```python
class chatbot_state(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
```

The state holds the full conversation as a list of messages. The `add_messages` reducer ensures that every new message is **appended** to the list instead of replacing it — that's how the model sees the whole history each turn.

### 2. Single chat node

```python
def chat_node(state: chatbot_state):
    message = state["message"]
    response = model.invoke(message)
    return {"message": response}
```

It reads the accumulated messages and invokes the LLM. The reply is returned and merged back into state.

### 3. Building the graph

```python
graph = StateGraph(chatbot_state)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
```

A simple 1-node pipeline: `START → chat_node → END`.

### 4. ⭐ SQLite checkpointer — persistence on disk

```python
con = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=con)
workflow = graph.compile(checkpointer=checkpoint)
```

This is the whole point of the folder. The compiled `workflow` now writes every checkpoint to `chatbot.db` on disk. Close the app, reopen it later, and the conversation is still there.

### 5. Asking the bot to recall its own history

```python
config = {"configurable": {"thread_id": "2"}}
res = workflow.invoke({"message": "what was the third last user message?"}, config=config)
for message in res["message"]:
    print(message.content)
```

Because the full message history for thread `"2"` is checkpointed in SQLite, the model receives all prior messages and can answer questions about its own past conversation — the classic test of a stateful chatbot.

## 📂 Files

| File | Purpose |
|---|---|
| `main.py` | Chatbot workflow compiled with `SqliteSaver`; tests recall of past messages |
| `config.py` | Groq model setup (`openai/gpt-oss-20b`) using `groq_api_key` |
| `chatbot.db` | The SQLite database file holding all checkpoints (generated at runtime) |
| `.env` | Groq API key (gitignored — never commit) |

## 🚀 Running It

```bash
# 1. Create .env with your Groq key
groq_api_key=your_key_here

# 2. Install dependencies
pip install langgraph langchain-groq python-dotenv

# 3. Run
python main.py
```

After running, inspect the database:

```bash
# With the sqlite3 CLI
sqlite3 chatbot.db ".tables"
sqlite3 chatbot.db "SELECT * FROM checkpoints;"
```

## 🔑 Key Takeaways

1. `SqliteSaver` gives **on-disk persistence** — memory survives restarts, with **no database server** required.
2. `sqlite3.connect(..., check_same_thread=False)` + `SqliteSaver(conn=con)` is the one-liner that turns a graph stateful *and* durable.
3. The `thread_id` mechanism works exactly like `InMemorySaver` — threads still isolate conversations.
4. Upgrade path: start with `InMemorySaver` (learn), move to `SqliteSaver` (local persistence), then `PostgresSaver` (scaled production).
