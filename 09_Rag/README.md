# 09_Rag — Retrieval-Augmented Generation (RAG)

This folder contains my RAG practice implementation using LangChain and LangGraph with Groq.

## 📁 Files in This Folder

| File | What I Practiced |
|---|---|
| `Rag_pipline.ipynb` | End-to-end RAG workflow: document loading → chunking → embeddings → vector store → retrieval → generation |
| `config.py` | Groq model setup (`llama-3.1-8b-instant`), OpenAI embeddings, Chroma vector store initialization |
| `README.md` | This file — practical reference of my RAG practice |

## 🛠️ My RAG Practice Workflow

Based on the notebook and config, here's what I actually did:

1. **Set up API keys** in `.env`:
   - `groq_api_key` = Groq API key for LLM inference
   - Optional `OPENAI_API_KEY` = for embeddings (if not using Groq embeddings)

2. **Installed dependencies**:
   ```bash
   pip install langchain langchain-groq langchain-community chroma-db sentence-transformers
   ```

3. **Created the RAG chain** (in `Rag_pipline.ipynb`):
   - Loaded documents from `knowledge_base/` folder
   - Chunked text with `RecursiveCharacterTextSplitter` (chunk_size=500, chunk_overlap=50)
   - Generated embeddings using OpenAIEmbeddings (or Groq-compatible)
   - Indexed into Chroma vector store
   - Built a retriever (top-k=3)
   - Created a prompt template with context + question
   - Composed the chain using LangChain Expression Language (`|`):
     ```python
     rag_chain = (
         {"context": retriever, "question": RunnablePassthrough()}
         | ChatPromptTemplate.from_template(
           "Answer using ONLY the context below:\n\nContext: {context}\n\nQuestion: {question}"
         )
         | ChatGroq(model="llama-3.1-8b-instant")
         | StrOutputParser()
     )
     ```

4. **Tested it** by asking questions about my knowledge base, seeing the model retrieve relevant chunks and answer grounded in those passages.

## 🔑 Key Configuration (`config.py`)

```python
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("groq_api_key")
model = ChatGroq(model='llama-3.1-8b-instant', api_key=api_key)

embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

## 🚀 How to Run My Practice

```bash
# 1. Set API key
groq_api_key=your_key_here

# 2. Install deps
pip install langchain langchain-groq langchain-community chroma-db sentence-transformers

# 3. Add knowledge base files to `knowledge_base/` folder
# (PDFs, CSVs, or .txt files)

# 4. Run the notebook or script
jupyter notebook Rag_pipline.ipynb
# #or#
python main.py
```

## 💡 What I Learned

- Embedding quality matters most — bad embeddings = poor retrieval
- Chunk size affects answer quality (500 tokens with 50 overlap worked well for my docs)
- The `RunnablePassthrough` pattern `|` is super useful for wiring retrieval + generation
- Groq's `llama-3.1-8b-instant` works well for RAG prototyping
- Vector store choice (Chroma for local prototyping) impacts performance

## 📬 References & Inspiration

- LangChain RAG Tutorial (official docs)
- Groq LLM integration patterns
- Chroma vector store quickstart
- Recursive character text splitting best practices