# AgenticAI Using LangGraph

A hands-on collection of agentic AI workflows built with **LangGraph** — from deterministic state machines to LLM-powered autonomous agents.

## 📦 Contents

| Project | Description | Highlights |
|---|---|---|
| [`00_Bmi_Cal_Workflow`](./00_Bmi_Cal_Workflow) | BMI calculator as a state graph | Deterministic workflow, `StateGraph` basics, 2-node pipeline |
| [`01_Simple_LLM_based`](./01_Simple_LLM_based) | LLM-powered Q&A with Groq & Llama 3.1 | External LLM integration, single-node graph, prompt engineering |
| [`02_Sequential_Workflow`](./02_Sequential_Workflow) | Prompt chaining — outline → blog → scoring | Sequential `StateGraph`, prompt injection, 3-node pipeline |
| [`03_Parallel_workflow`](./03_Parallel_workflow) | HR screening pipeline & essay evaluation | Parallel nodes, structured output, multi-angle evaluation |
| [`04_Conditonal_workflow`](./04_Conditonal_workflow) | Sentiment-based review routing | `add_conditional_edges`, branching, Pydantic structured output |
| [`05_Iterative_Workflow`](./05_Iterative_Workflow) | Generate → evaluate → optimize loop | Iterative cycle, LLM-as-a-judge, max-iteration guard |

## 🧠 What is LangGraph?

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework for building stateful, multi-actor applications with LLMs. It extends LangChain by modeling agent workflows as **directed graphs** where:

- **Nodes** represent computation steps (e.g., tool calls, LLM invocations, deterministic logic).
- **Edges** define control flow between nodes.
- **State** is a shared, typed object passed through the graph.

This approach enables complex agent behaviours — branching, looping, human-in-the-loop — that are difficult to express with simple chains.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook / VS Code with notebook support

### Installation

```bash
# Clone the repository
git clone https://github.com/huzaifaqazi/AgenticAI-Using-LangGraph.git
cd AgenticAI-Using-LangGraph

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install core dependencies
pip install langgraph typing-extensions
```

For LLM-based notebooks you will also need:

```bash
pip install langchain-groq
```

### Running a Notebook

```bash
jupyter notebook
```

Then navigate to any project folder and open the `.ipynb` file.

## 📁 Project Structure

```
AgenticAI-Using-LangGraph/
├── 00_Bmi_Cal_Workflow/
│   └── 00_bmi_cal_workflow.ipynb
├── 01_Simple_LLM_based/
│   └── Simple_LLm_based_workflow.ipynb
├── 02_Sequential_Workflow/
│   └── Sequential_workflow.ipynb
├── 03_Parallel_workflow/
│   └── hr_screening_pipeline.py
├── 04_Conditonal_workflow/
│   └── main.py
├── 05_Iterative_Workflow/
│   └── main.py
└── README.md
```

## 🤝 Contributing

Contributions are welcome! If you have an idea for a new agent workflow or an improvement to an existing one, feel free to open an issue or submit a pull request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
