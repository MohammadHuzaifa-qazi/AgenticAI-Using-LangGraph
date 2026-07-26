# Parallel Workflow

## 📌 What is a Parallel Workflow?

A **parallel workflow** is an execution model where multiple independent tasks run **simultaneously** rather than one after another. In traditional sequential processing, Task B waits for Task A to finish, even if they don't depend on each other. This wastes time.

In a parallel workflow, all independent tasks execute concurrently, and their results are merged at a central aggregation point. This is especially powerful in LLM-based systems where multiple evaluations, analyses, or generations can happen in parallel — each with its own focused prompt — and the combined output drives a final decision.

### Real-World Analogy

Imagine a hiring committee with three specialists:
- A **technical lead** evaluates the candidate's skills
- An **HR manager** assesses communication quality
- A **culture officer** judges team fit

They all review the application **at the same time**, then meet to decide. This is exactly what a parallel workflow does.

### In LangGraph

LangGraph achieves parallelism by routing multiple edges from `START` to different **nodes**. Each node runs in its own thread. When all complete, their outputs merge into the shared **state**, which flows into a final aggregation node.

```
                    ┌──────────────┐
                    │   Node A     │
                    └──────┬───────┘
                    ┌──────────────┐
START ──────────────│   Node B     │───→ Final Node → END
                    └──────────────┘
                    ┌──────────────┐
                    │   Node C     │
                    └──────────────┘
```

---

## 🎯 What Problem Does This Solve?

### The Problem

When HR teams screen hundreds of job applications, they must evaluate each candidate on multiple dimensions:
- **Skills** — Do they match the job description?
- **Communication** — Can they express themselves clearly?
- **Culture fit** — Will they thrive in the team?

Doing this manually is:
- ⏱ **Slow** — 5–10 minutes per resume, hours for a batch
- 🧠 **Inconsistent** — Different evaluators have different standards
- 🎭 **Biased** — Subjective impressions can override objective criteria

### Our Solution

This project automates candidate screening using an **LLM-powered parallel workflow**. Instead of sequential checks, all evaluations run concurrently. Each node has a focused prompt and a **structured output schema** (via Pydantic), ensuring consistent, typed results. A final node aggregates the scores and produces a clear, explainable recommendation.

---

## 👥 Who Is This For?

| Role | How They Benefit |
|------|------------------|
| **HR Professionals** | Reduce manual screening time from hours to seconds |
| **Hiring Managers** | Get consistent, criteria-based candidate evaluations |
| **Engineering Teams** | Automate initial screening for technical roles |
| **AI/ML Developers** | Learn how to build parallel LangGraph workflows |

---

## 🏗 Why Was This Built?

This project was built to demonstrate **parallel execution in LangGraph** — a concept that most tutorials skip. The key lessons are:

1. **Parallel nodes** — Multiple nodes can run simultaneously from `START`
2. **State merging** — Concurrent writes to different state keys don't conflict
3. **Structured output** — Pydantic schemas force the LLM to return predictable data
4. **Deterministic aggregation** — The final node applies business logic on LLM outputs

The HR use case was chosen because it's **relatable** — everyone understands the hiring process — and **practical** for showcasing multi-angle evaluation.

---

## 🔁 How This System Works

### Architecture

```
                    ┌─────────────────────┐
                    │  technical_score    │ ← skills + job_description → score (0-100)
                    └──────────┬──────────┘
                    ┌──────────────────────────┐
START ──────────────│ communication_quality    │ ← why_hire_me → Excellent / Good / Need_improvement
                    └──────────┬───────────────┘
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

---

## 🧠 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Parallel execution** | 3 nodes from `START` run simultaneously via separate edges |
| **State merging** | All parallel outputs converge at `final_recommendation` |
| **Structured output** | Pydantic `BaseModel` ensures LLM returns typed, predictable data |
| **Reducer-free design** | Each node writes to different state keys — no conflict on concurrent writes |
| **Deterministic aggregation** | Final node applies hard business logic on aggregated LLM outputs |

---

## 🗂 Files

| File | Purpose |
|------|---------|
| `hr_screening_pipeline.py` | Main HR parallel screening workflow (skills + communication + culture → recommendation) |
| `essay_evaluation.py` | Alternative parallel workflow — essay evaluation with clarity, analysis, language |
| `resume_checking_system.py` | Resume skill + experience analysis (sequential, precursor to this project) |
| `.env` | Groq API key configuration |

---

## 🚀 Run

```bash
python hr_screening_pipeline.py
```

---

## 📊 Sample Output

```
The Score based on the Skills:80
The communication quality is :Good
The culture fit is :Strong
The final recommendation is :strong Hire
```
