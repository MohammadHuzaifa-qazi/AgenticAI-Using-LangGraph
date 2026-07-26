# Parallel Workflow — HR Screening Pipeline

## 📌 What is a Parallel Workflow?

A parallel workflow runs multiple independent tasks **simultaneously** rather than sequentially. In LangGraph, this is achieved by routing multiple edges from `START` to different nodes — each node executes in its own thread, then all results converge at a final aggregation node.

This pattern is ideal when:
- You have multiple independent evaluations to perform on the same input
- You want to minimise total latency by running tasks concurrently
- Each evaluation has its own focused prompt and criteria

## 🔁 How This System Works

### Architecture

```
                    ┌─────────────────────┐
                    │  technical_score    │ ← skills + job_description → score (0-100)
                    └──────────┬──────────┘
                    ┌─────────────────────┐
START ──────────────│ communication_quality │ ← why_hire_me → Excellent / Good / Need_improvement
                    └──────────┬──────────┘
                    ┌─────────────────────┐
                    │    culture_fit      │ ← why_hire_me → Strong / Moderate / Weak
                    └──────────┬──────────┘
                               │
                    ┌──────────▼───────────┐
                    │ final_recommendation │ → Strong Hire / Consider / Reject
                    └──────────┬───────────┘
                              END
```

### Node Breakdown

| Node | Input | Evaluation Criteria | Output |
|------|-------|-------------------|--------|
| `technical_score` | skills, job_description | Skill match with job requirements | Score (0–100) |
| `communication_quality` | why_hire_me | Clarity, tone, grammar, confidence | Excellent / Good / Need_improvement |
| `culture_fit` | why_hire_me | Teamwork, values, attitude, adaptability | Strong / Moderate / Weak |
| `final_recommendation` | All 3 above | Weighted decision logic | Strong Hire / Consider / Reject |

### Decision Logic

```
if score ≥ 80 AND culture_fit == "Strong"  → "Strong Hire"
elif score ≥ 70 AND communication == "Excellent" → "Consider"
else → "Reject"
```

## 🧠 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Parallel execution** | 3 nodes from `START` run simultaneously via separate edges |
| **State merging** | All parallel outputs converge at `final_recommendation` |
| **Structured output** | Pydantic `BaseModel` ensures LLM returns typed, predictable data |
| **Reducer-free design** | Each node writes to different state keys — no conflict on concurrent writes |
| **Conditional routing** (extensible) | Final node applies deterministic business logic on aggregated state |

## 🗂 Files

| File | Purpose |
|------|---------|
| `hr_screening_pipeline.py` | Main HR parallel screening workflow |
| `essay_evaluation.py` | Alternative parallel workflow — essay evaluation with clarity, analysis, language |
| `resume_checking_system.py` | Resume skill + experience analysis (sequential) |
| `.env` | Groq API key configuration |

## 🚀 Run

```bash
python hr_screening_pipeline.py
```

## 📊 Sample Output

```
The Score based on the Skills:80
The communication quality is :Good
The culture fit is :Strong
The final recommendation is :strong Hire
```
