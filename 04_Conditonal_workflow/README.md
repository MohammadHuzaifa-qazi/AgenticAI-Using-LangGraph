# Conditional Workflow — Smart Routing in LangGraph

## 📌 What is a Conditional Workflow?

A **conditional workflow** (or conditional routing) lets the graph **decide which path to take at runtime** based on the current state. Instead of always flowing through the same sequence of nodes, a conditional edge inspects the state and routes execution to the appropriate branch.

Think of it like a customer-support triage system: the first agent classifies your issue (refund, tech problem, complaint), and then your call is routed to the right department. Each department only handles its own type of issue.

## 🔀 Why Use Conditional Routing?

| Benefit | Explanation |
|---|---|
| ✅ **Context-aware** | The system adapts its behaviour based on the actual input, not a fixed template |
| ✅ **Efficiency** | Only runs the nodes that are relevant to the current input — saves time & tokens |
| ✅ **Huma-assist** | LLM can be the decision-maker, generating the condition itself |
| ✅ **Specialized responses** | Different branches can use completely different prompts and schemas |
| ✅ **Matches real workflows** | Many real systems (support, moderation, review) naturally branch on conditions |

## 🧠 Key LangGraph Concept: `add_conditional_edges`

The heart of this pattern is:

```python
graph.add_conditional_edges(
    "Find_Sentiment",     # source node
    check_sentiment,      # routing function → returns a branch name
    {
        "positive": "positive_sentiment",   # route for positive review
        "negative": "run_diagnose",         # route for negative review
    },
)
```

The function `check_sentiment(state)` reads the `state["sentiment"]` value written by the LLM and returns one of the branch keys. LangGraph then follows the matching edge.

## 🏗 How This Example Works

### The Problem: Review Handling

You receive a customer review. Depending on whether it is **positive** or **negative**, you should respond completely differently. One fixed response won't do.

### The Graph

```
            ┌──────────────────────┐
            │ start/Find_Sentiment │  ← LLM classifies: positive / negative
            └──────────┬───────────┘
                       │ (conditional)
              ┌────────┴────────┐
        positive│              │negative
              ▼                ▼
   ┌──────────────────┐   ┌────────────────────┐
   │ positive_sentiment│   │ run_diagnose        │  ← structured issues, tone, urgency
   └────────┬─────────┘   └─────────┬──────────┘
            │                       │
            │                       ▼
            │                ┌─────────────────┐
            └────────────────▶   negative_sentiment │
                            └────────┬────────┘
                                      │
                                      ▼
                                    [END]
```

### Managed state (`review_state`)

| Field | Type | Filled by |
|---|---|---|
| `review` | `str` | input — the customer review |
| `sentiment` | `"positive" / "negative"` | node id: `Find_Sentiment` |
| `diagnose` | `dict` | `run_diagnose` (used only on the negative branch) |
| `result` | `str` | final output message |

### Step by step

1. **`Find_Sentiment`** — the LLM reads the review and classifies the sentiment. It writes `state["sentiment"]`.
2. **Conditional edge** — the state is inspected:
   - → `positive`: the review is happy, so we route to **`positive_sentiment`**.
   - → `negative`: the review has issues, so we route to **`run_diagnose`**.
3. **Negative branch** — `run_diagnose` uses a Pydantic schema (`diagnose_schema`) to extract `issue_type`, `tone`, and `urgency`, then `negative_sentiment` builds an empathetic resolution message.
4. **Positive branch** — `positive_sentiment` generates a warm thank-you and asks for feedback.
5. Both branches finish at **`END`**.

## 🔧 Structured Output (Pydantic)

Conditional branches often need the LLM to return *validated, typed* data so the routing logic can reliably act on it. This uses Pydantic schemas:

```python
class review_schema(BaseModel):
    sentiment: Literal["positive", "negative"]

model = ChatGroq(...).with_structured_output(review_schema)
```

This guarantees the routing function gets clean, predictable values — no guessing from raw text.

## 🚀 Run

```bash
python main.py
```

## 📌 Files

| File | Purpose |
|------|---------|
| `main.py` | The full conditional workflow (sentiment → positive/negative routing) |
| `.env` | Groq API key (not committed) |