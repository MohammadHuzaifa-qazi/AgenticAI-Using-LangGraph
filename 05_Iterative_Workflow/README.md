# Iterative Workflow — Generate → Evaluate → Optimize Loop

## 📌 What is an Iterative Workflow?

An **iterative workflow** is a graph where execution **loops back** to an earlier node instead of flowing straight to the end. This lets the system **refine its output** repeatedly until a quality condition is satisfied — or a maximum number of rounds (cap) is reached.

This is commonly called the **LLM-as-a-judge** or **self-refinement** pattern: a generator proposes, an evaluator grades it, and an optimizer improves it, looping until the evaluator approves.

## 🔁 Why Use an Iterative Workflow?

| Benefit | Explanation |
|---|---|
| ✅ **Higher quality output** | Output is refined over multiple rounds rather than generated once |
| ✅ **Self-correction** | The system catches and fixes its own mistakes using feedback |
| ✅ **Guard rails** | A `max_iterations` counter prevents infinite / runaway loops |
| ✅ **Testable/measurable** | A structured evaluator gives explicit feedback and a verdict |
| ✅ **Broadly applicable** | Content writing, code review, translation, summarization, negotiation |

## 🧠 The Core Pattern: A Cycle in the Graph

Loops in LangGraph are created by a **conditional edge that points back to an earlier node**. The condition function decides whether to *continue iterating* or *exit*.

```python
graph.add_conditional_edges(
    "Evaluate_post",
    evaluation_decision,              # returns a route key
    {
        "Approved": END,             # done → exit
        "Need_improved": "optimizer" # not good enough → loop back
    },
)
```

### The termination guard

An infinite loop is dangerous. The state carries an `iteration` counter, and the routing function stops the loop when the cap is hit:

```python
def evaluation_decision(state):
    if state["iteration"] >= state["max_iterations"]:
        return "Approved"          # force exit at cap
    return state["evaluator"]      # "Approved" or "Need_improved"
```

## 🏗 How This Example Works

### The Problem: Write a Great Twitter Post

Write a post about *"How we celebrate Independence Day in Pakistan"*. A single generation may be too long, off-tone, or miss hashtags. So we loop: **generate → evaluate → optimize** until it passes quality checks.

### Managed state (`Post_State`)

| Field | Type | Purpose |
|---|---|---|
| `topic` | `str` | input — the subject of the post |
| `post_text` | `str` | the current draft (updated by generator & optimizer) |
| `evaluator` | `"Approved" / "Need_improved"` | verdict from the evaluator |
| `feedback` | `str` | the evaluator's critique, fed to the optimizer |
| `iteration` | `int` | loop counter (starts at 1) |
| `max_iterations` | `int` | safety cap to prevent infinite loops |

### The graph

```
 START
   │
   ▼
 Generate_post ──► Evaluate_post
                        │   ▲
          ┌─────────────┘   │
          |   Need_improved │ (loop back)
          ▼                 │ optimizer
        optimizer ──────────┘
          │
          ▼
        [END]
```

### Step by step

1. **`Generate_post`** — the LLM (persona = "social media expert") writes the initial tweet from the topic.
2. **`Evaluate_post`** — the *judge* LLM checks the draft against rules (character count, tone, hashtags, language) and returns `evaluator` verdict + `feedback` through a Pydantic schema.
3. **Conditional edge** (`evaluation_decision`):
   - `Approved` → **`END`**
   - `Need_improved` and `iteration < max_iterations` → **`optimizer`**
4. **`Optimizer`** — rewrites `post_text` using the critic's `feedback` and increments `iteration` (`iteration + 1`).
5. **Loop back** to `Evaluate_post` for another round.
6. Loop ends when the evaluator approves or `max_iterations` is reached.

> In the demo: `iteration = 1`, `max_iterations = 2` → it will loop at most twice, then stop.

## 📂 Files

| File | Purpose |
|------|---------|
| `config.py` | Centralized LLM setup (`ChatGroq`) shared by the graph |
| `main.py` | The generate → evaluate → optimize iterative workflow |
| `.env` | Groq API key (not committed) |

## 🚀 Run

```bash
python main.py
```

---

> 💡 **Tip:** This pattern powers AI content-refinement tools, automated code review, and AI agents that "think harder" until their answer passes their own quality bar.