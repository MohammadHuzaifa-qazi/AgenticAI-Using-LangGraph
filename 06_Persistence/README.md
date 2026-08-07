# 06_Persistence — Checkpointers, State History & Time Travel

Persistence (checkpointing) is the feature that turns a **stateless** LangGraph graph into a **stateful** one. It lets LangGraph remember the state of every node execution, save it to a checkpointer, and — most importantly — lets you **resume**, **rewind**, and **fork** a conversation later.

## 📖 Topic Explained in Detail

### Why do we need Persistence?

Without a checkpointer, every `graph.invoke()` call starts from a completely fresh state. The graph forgets everything as soon as it finishes. In real agent applications we need memory:

- **Multi-turn conversations** — the bot must remember what the user said earlier.
- **Human-in-the-loop** — pause a workflow, ask a human for approval, then continue.
- **Fault recovery** — if a step crashes, resume from the last checkpoint instead of restarting.
- **Auditing / debugging** — inspect exactly what state each node produced.
- **Time travel** — rewind to a past checkpoint, change the input, and replay the future.

### What is a Checkpointer?

A **checkpointer** is a storage backend that saves a snapshot of the graph state **after every node** finishes. Each snapshot is called a **checkpoint** and is uniquely identified by a **`thread_id`** (and an optional **`checkpoint_id`**).

LangGraph provides several checkpointers:

| Checkpointer | Storage | Use case |
|---|---|---|
| `InMemorySaver` | RAM | Experiments, tutorials, stateless quick tests |
| `SqliteSaver` | SQLite file | Persistent local storage |
| `PostgresSaver` | PostgreSQL | Production, multi-process / multi-user apps |
| `MongoDB` / custom saver | Various | Advanced custom needs |

This project uses `InMemorySaver` — data lives only while the process runs, which is perfect for learning the concepts.

### Threads — `thread_id`

A **thread** is a logical "conversation line". You pass it via the runtime `config`:

```python
config = {"configurable": {"thread_id": "1"}}
```

- Same `thread_id` → same conversation memory (checkpoints accumulate).
- Different `thread_id` → isolated conversations that don't interfere.

### `get_state` vs `get_state_history`

- **`get_state(config)`** — returns the **latest** checkpoint of a thread: the full state values (`.values`) plus metadata like the next node.
- **`get_state_history(config)`** — returns a **list of ALL checkpoints** in the thread, oldest → newest. This is the raw material for time travel.

### Time Travel

Because every node is checkpointed, you can:

1. List the history of a thread.
2. Pick a past **`checkpoint_id`**.
3. `invoke(None, config_with_checkpoint_id)` to **replay** from that point.
4. `update_state(config, {...})` to **edit** the state at that checkpoint (fork), then replay the future from the fork.

This is like a Git history for your agent conversation.

## 🧠 Code Concepts Explained in Detail

### 1. Structured Output with Pydantic

```python
class str_schema(BaseModel):
    question: str
    city: str
    population: int

res = model.with_structured_output(str_schema)
```

Instead of parsing free text, we ask the LLM to return a **typed Pydantic object**. `.with_structured_output()` guarantees `res.city` and `res.population` exist as real fields.

### 2. Typed State

```python
class simple_state(TypedDict):
    question: str
    city: str
    population: str
```

The state is a `TypedDict` that flows through the whole graph. Each node reads what it needs and returns a dict of the fields it wants to update.

### 3. Nodes — reading state, returning updates

```python
def city(state: simple_state):
    question = state["question"]
    prompt = f"Based on the question: '{question}', please provide the name of the city."
    response = res.invoke(prompt)
    return {"city": response.city}
```

**Key pattern:** a node **reads** from `state` and returns **only the partial update** (a dict). LangGraph merges that update back into the shared state for the next node.

The same logic is repeated in `population`, which reads the `city` written by the previous node:

```python
def population(state: simple_state):
    city_name = state["city"]
    prompt = f"Based on the city: '{city_name}', please provide the population of that city."
    response = res.invoke(prompt)
    return {"population": response.population}
```

This is a classic **pipeline**: question → city → population.

### 4. Building the Graph

```python
graph = StateGraph(simple_state)
graph.add_node("city", city)
graph.add_node("population", population)
graph.add_edge(START, "city")
graph.add_edge("city", "population")
graph.add_edge("population", END)
```

- `StateGraph` wires the typed state to the nodes.
- `add_node` registers a node with a name.
- `add_edge` connects nodes; `START` is the entry point, `END` terminates the graph.

### 5. ⭐ Enabling Persistence — the core idea

```python
checkpoint = InMemorySaver()
workflow = graph.compile(checkpointer=checkpoint)
```

The graph is only compiled **once**, and the **checkpointer is attached at compile time**. From now on, the compiled `workflow` records a checkpoint after every node execution.

### 6. Isolated threads

```python
config  = {"configurable": {"thread_id": "1"}}
config2 = {"configurable": {"thread_id": "2"}}

output = workflow.invoke(initial_state, config=config2)
```

Thread `"1"` and thread `"2"` are **fully isolated**. Even though it is the same compiled graph, each thread keeps its own checkpoint chain.

### 7. Inspecting state and history

```python
print(workflow.get_state(config2))          # latest checkpoint of thread 2
print(list(workflow.get_state_history(config2)))  # every checkpoint in thread 2
```

- `get_state` → current state values + metadata.
- `get_state_history` → the full timeline of checkpoints, which enables time travel.

### 8. Time Travel — replay and fork (from the notebook)

The notebook defines a second graph (`joke` → `explaination`) and demonstrates the full time-travel workflow:

```python
# 1. Get the full checkpoint history of the thread
list(workflow.get_state_history({"configurable": {"thread_id": "1"}}))

# 2. Read the state at a SPECIFIC past checkpoint_id
workflow.get_state({"configurable": {"thread_id": 1, "checkpoint_id": "..."}})

# 3. REPLAY from that checkpoint (None = no new input, just rerun from there)
output = workflow.invoke(None, {"configurable": {"thread_id": 1, "checkpoint_id": "..."}})

# 4. FORK: update the state at a past checkpoint (change topic → "Laptop")
workflow.update_state(
    {"configurable": {"thread_id": 1, "checkpoint_id": "...", "checkpoint_ns": ""}},
    {"topic": "Laptop"},
)

# 5. Re-invoke — the graph continues from the fork with the new topic
res = workflow.invoke(None, {"configurable": {"thread_id": 1}})
```

**What happens here:**

| Step | What it does |
|---|---|
| `get_state_history` | Lists checkpoints so you can find a `checkpoint_id` to jump to |
| `get_state(checkpoint_id)` | Peers into the state **at that moment** |
| `invoke(None, checkpoint_id)` | **Replays** the graph from that past checkpoint |
| `update_state(checkpoint_id)` | **Rewrites** the state at a past checkpoint (creates a fork in the timeline) |
| `invoke(None)` on the fork | Regenerates the future using the edited topic (`Laptop`) |

This is the "Git of conversations" — you can see history, checkout an old commit, edit it, and continue from there.

## 🛡️ Fault Tolerance — recovering from interruptions

The `fault_tolerance.ipynb` notebook shows the second huge benefit of checkpoints: **a graph that crashed or was interrupted can resume from exactly where it stopped** — no node is run twice, and none is skipped.

### How it works

`step2` deliberately sleeps for 20 seconds to simulate a slow/long task:

```python
def step2(state: fault_state):
    print("Step2 is loading........")
    time.sleep(20)
    print("Step2 is successfully........")
    return {"step2": "Done"}
```

If you interrupt the run (`KeyboardInterrupt`) during that sleep, the run dies mid-flight. Without a checkpointer that would be a disaster — you'd restart from zero. With `InMemorySaver`, every completed node is already saved.

### Resuming after a failure

```python
# 1. See the saved state despite the interruption
workflow.get_state(config)

# 2. See the whole timeline
list(workflow.get_state_history(config))

# 3. Resume: invoke with None → continues from the last checkpoint, NOT from START
output2 = workflow.invoke(None, config)
```

The magic is in step 3: passing `None` as the input means "no new input, just continue the existing thread". LangGraph loads the last checkpoint, sees that `step1` already finished, and runs **only** the remaining work (`step2`). After resuming, `get_state` shows a complete state and the history shows the new checkpoints appended.

> **Key idea:** a checkpointer turns a crash into a "pause". The graph remembers how far it got, so retry/continuation is cheap and idempotent — the core of production fault tolerance, retries, and human-in-the-loop approvals.

## 📂 Files

| File | Purpose |
|---|---|
| `checkpointers1.ipynb` | Full walkthrough: persistence, threads, state history, time travel |
| `fault_tolerance.ipynb` | Fault tolerance: interrupt a run, then resume from the last checkpoint |
| `main.py` | Runnable script: two isolated threads + state/history inspection |
| `test_history.py` | Compact version: invoke, read latest state, count history entries |
| `config.py` | Shared model setup (`ChatGroq` + `groq_api_key`) |

## 🚀 Running It

```bash
# 1. Create .env with your Groq key
groq_api_key=your_key_here

# 2. Install dependencies
pip install langgraph langchain-groq python-dotenv

# 3. Run
python main.py          # two threads + history inspection
python test_history.py  # compact state/history test
```

## 🔑 Key Takeaways

1. **Compile with a checkpointer** → the graph becomes stateful: `graph.compile(checkpointer=checkpoint)`.
2. **Threads isolate conversations** → same `thread_id` = shared memory.
3. **`get_state`** shows the latest snapshot; **`get_state_history`** shows the full timeline.
4. **Time travel** = `checkpoint_id` + `invoke(None, ...)` (replay) or `update_state(...)` (fork).
5. Persistence is the foundation of **human-in-the-loop**, **fault tolerance**, and **multi-turn memory** in production agents.
