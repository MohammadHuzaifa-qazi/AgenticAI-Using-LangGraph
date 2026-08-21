# 09_Rag — Retrieval-Augmented Generation (RAG)

This folder demonstrates **Retrieval-Augmented Generation (RAG)** — a pattern that combines information retrieval with LLM generation to ground responses in external knowledge.

## 📖 Topic Explained in Detail

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM responses by first retrieving relevant documents or passages from an external knowledge source, then passing those retrieved chunks as context to the LLM along with the user prompt.

Without RAG, an LLM relies solely on its internal training weights — which have a cutoff date and cannot include:
- Private or organization-specific data
- Very recent events
- Domain-specific technical docs
- Large corpora that would exceed the model's context window

RAG solves this by: **`Retrieve → Augment → Generate`**.

### Why use RAG? (Benefits)

| Problem | RAG Solution |
|---|---|
| Model hallucinates or makes things up | Retrieval provides **grounded, factual evidence** from real documents |
| Knowledge is stale (cutoff date) | Index fresh documents and query them at runtime |
| Private/company data can't be used | Store internal docs in a vector database; never leave your network |
| Context window too small for long docs | Retrieve only the **relevant snippets** (e.g., 300–500 tokens) instead of the whole book |
| User asks domain-specific questions | Use an embeddings model to find the most semantically similar chunks |

**Key advantages:**
- No fine-tuning required — just prompt engineering + retrieval
- Updates to knowledge base are instant (just re-index)
- Works with any LLM (Groq, OpenAI, Anthropic, etc.)
- Can combine multiple retrieval strategies (semantic, keyword, hybrid)

### The RAG Pipeline (Step-by-Step)

A typical RAG pipeline has these stages:

1. **Ingest / Load** — Load documents (PDFs, web pages, databases, APIs)
2. **Chunking** — Split long documents into smaller overlapping chunks (e.g., 512 tokens with 50-token overlap) so context is preserved
3. **Embedding** — Run each chunk through an embedding model (e.g., `all-MiniLM-L6-v2`, `text-embedding-3-large`) to produce vector representations
4. **Indexing** — Store chunk vectors in a **vector database** (Chroma, Pinecone, Weaviate, Qdrant, Milvus) with metadata (source, page number, date)
5. **Retrieval** — When a user query arrives:
   - Embed the query using the same embedding model
   - Perform **similarity search** (typically cosine similarity) against the stored vectors
   - Return the top-k most similar chunks (k = 3–5 is common)
6. **Augmentation** — Prepend the retrieved chunks to the user prompt, forming an "augmented" prompt:
   ```
   [BEGIN CONTEXT]
   {chunk_1}
   {chunk_2}
   {chunk_3}
   [END CONTEXT]

   User question: {original_query}
   ```
7. **Generation** — Pass the augmented prompt to the LLM (via LangChain, LlamaIndex, or direct API call). The model cites the retrieved passages when answering.

### Where is RAG used?

| Use Case | Example |
|---|---|
| **Internal knowledge chatbot** | Employees ask about HR policies, onboarding docs, SOPs |
| **Customer support** | Bot answers from product manuals, troubleshooting guides |
| **Legal / medical assistant** | Queries over case law, medical textbooks (with proper disclaimers) |
| **Code assistant** | Retrieves from internal codebase, docs, wikis |
| **Research assistant** | Scholar over academic papers, arXiv, internal reports |
| **Enterprise search** | Find relevant doc snippets across Confluence, SharePoint, Notion |

### Key Components in LangChain / LangGraph

| Component | Purpose | Typical Library |
|---|---|---|
| `DocumentLoader` | Load raw files (PDF, txt, HTML) | `langchain_community.document_loaders` |
| `TextSplitter` | Chunk documents into manageable pieces | `RecursiveCharacterTextSplitter` |
| `Embeddings` | Convert text → vector embeddings | `sentence-transformers`, `openai` embeddings |
| `VectorStore` | Store embeddings + fast similarity search | Chroma, Pinecone, Qdrant, Weaviate |
| `Retriever` | Wrap vector store search + return top-k docs | `VectorStoreRetriever` |
| `RunnablePassthrough` / `assign` | Wire retrieval + generation together | Lang Expression Language (`|`) |
| `ChatPromptTemplate` | Build the augmented prompt (context + question) | `langchain_core.prompts` |

### Code Overview (Typical LangChain RAG)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# 1. Load
loader = PyPDFLoader("docs/company_policy.pdf")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed + Index
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Prompt template
template = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

    Context:
    {context}

    Question: {question}
    """
)

# 6. Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | ChatGroq(model="llama-3.1-8b-instant")
    | StrOutputParser()
)

result = rag_chain.invoke("What is our vacation policy?")
print(result)
```

## 📂 Files in This Folder

| File | Purpose |
|---|---|
| `config.py` | Groq model setup + embeddings + vectorstore initialization |
| `main.py` | End-to-end RAG chain: load → chunk → embed → retrieve → generate |
| `knowledge_base/` | Sample documents (PDFs, CSVs, text files) used for retrieval |
| `.env` | API keys (`groq_api_key`, optional `OPENAI_API_KEY` for embeddings) |
| `README.md` | This file — topic explanation + pipeline diagram |

## 🚀 Running the RAG Demo

```bash
# 1. Set up your API keys in .env
groq_api_key=gsk_...
# (Optional) OPENAI_API_KEY=sk-...  # if using OpenAI embeddings

# 2. Install dependencies
pip install langchain langchain-groq langchain-community chroma-db sentence-transformers

# 3. Add sample knowledge base files (or point to your own)
# Place PDFs, CSVs, or .txt files in the `knowledge_base/` folder

# 4. Run the main script
python main.py
```

The script will:
1. Load and chunk the documents
2. Create a Chroma vector store
3. Enter an interactive loop where you can ask questions
4. For each question, it retrieves the top-3 relevant chunks and asks the LLM to answer using only those passages

## 🧠 Key Takeaways

1. **RAG = Retrieval + Augmentation + Generation** — it's a pipeline, not just a single prompt.
2. The **embedding model** and **retriever** are the most critical parts — bad embeddings = bad retrieval = wrong answers.
3. **Chunking strategy** greatly impacts quality — too large and the model gets distracted; too small and context is lost.
4. **Vector store choice** matters for performance — Chroma is great for local prototyping; Pinecone/Weaviate for production scale.
5. RAG separates *knowledge* from *model weights* — you can update the knowledge without touching the model.