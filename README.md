
# AskMyDocs 📚
### Your Documents. Your Questions. AI-Powered Answers.

AskMyDocs is an AI-powered document question-answering application that allows users to upload documents and ask questions about their content using natural language.

Built with Python, Streamlit, LangChain, OpenAI Embeddings, and Pinecone, the project demonstrates a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant document chunks and uses an LLM to generate contextual answers.

## ✨ Features

- **Document Upload** — Upload PDF, TXT, and DOCX files.
- **Document Processing** — Extract text and split documents into smaller chunks.
- **Semantic Search** — Convert document chunks into vector embeddings and store them in Pinecone.
- **RAG-Based Question Answering** — Retrieve relevant information to help the LLM answer questions based on uploaded documents.
- **Source References** — Display source document names alongside generated answers.
- **Interactive Chat Interface** — Ask follow-up questions through a Streamlit chat interface.
- **Chat Management** — Clear the current conversation when needed.
- **Readable Responses** — Render Markdown-formatted answers and mathematical expressions.

## 🏗️ Architecture

```text
                 User
                  |
                  v
          Streamlit Web App
                  |
          Upload a Document
                  |
                  v
           Document Loader
                  |
                  v
        Text Splitting / Chunking
          Chunk Size: 1000
         Chunk Overlap: 200
                  |
                  v
         OpenAI Embeddings
       text-embedding-3-small
                  |
                  v
            Pinecone DB
                  |
                  | Semantic Retrieval
                  v
           Relevant Chunks
                  |
                  v
            LLM + Prompt
                  |
                  v
       Contextual AI Response
                  |
                  v
         Answer + Sources
```

## 🔍 How It Works

AskMyDocs follows a Retrieval-Augmented Generation (RAG) workflow.

### 1. Document Ingestion

Users upload a supported document. The application extracts its text and prepares it for indexing.

### 2. Text Chunking

The extracted text is divided into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

| Parameter | Value |
|---|---:|
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | 1,536 |

Chunking helps the retrieval system identify relevant passages without processing an entire document as a single unit.

### 3. Embedding and Vector Storage

Each chunk is converted into a numerical vector using OpenAI Embeddings and stored in Pinecone for semantic retrieval.

### 4. Question Answering

When a user submits a question, the RAG pipeline retrieves relevant document chunks and passes them to the language model as context.

The LLM then generates an answer grounded in the retrieved content.

### 5. Response Display

The application displays the generated answer and available source document names in the chat interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| LangChain | RAG pipeline and document processing |
| OpenAI API | Embeddings and language-model integration |
| Pinecone | Vector database and semantic retrieval |
| RecursiveCharacterTextSplitter | Document chunking |
| PyPDF / document loaders | Document text extraction |
| python-dotenv | Environment variable management |

## 📁 Project Structure

```text
AskMyDocs/
├── main.py                 # Streamlit application
├── ingestion.py            # Document chunking and vector indexing
├── rag/
│   ├── __init__.py
│   ├── core.py             # Retrieval and LLM response generation
│   └── loader.py           # PDF, TXT, and DOCX loading
├── .env                    # API credentials (not committed)
├── .gitignore
├── pyproject.toml          # Project dependencies and configuration
└── README.md
```

*The structure above reflects the intended organization; adjust filenames if your repository differs.*

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- An OpenAI API key
- A Pinecone account and API key
- A Pinecone dense vector index compatible with the LangChain vector-store integration

### 1. Clone the Repository

```bash
git clone https://github.com/IttikornMafaka/AskMyDocs.git
cd AskMyDocs
```

Replace the repository URL with your actual GitHub repository URL if it differs.

### 2. Install Dependencies

If the project uses `uv`:

```bash
uv sync
```

Alternatively, create a virtual environment and install the dependencies listed in your project configuration.

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Add any additional environment variables required by your implementation.

**Security:** Never commit API keys or your `.env` file to GitHub.

### 4. Configure Pinecone

Create a dense vector index with the following settings when using the default embedding configuration:

- Index name: `askmydocs`
- Dimension: `1536`
- Metric: `cosine`

Ensure that `ingestion.py` and `rag/core.py` reference the same index.

The index must support the vector/record API used by `PineconeVectorStore`; a Pinecone Document Schema index requires a different integration.

### 5. Run the Application

```bash
uv run streamlit run main.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## 💡 Example Usage

1. Upload a PDF containing course notes or technical documentation.
2. Click **Process Document** to extract, chunk, embed, and index the content.
3. Enter a question related to the uploaded document.
4. Read the generated answer.
5. Expand the Sources section to inspect the referenced document names.

Example question:

> What is supervised learning, and how does a model make predictions for new inputs?

The application retrieves relevant passages and uses them as context for its response.

## 🎯 Learning Objectives

This project provides hands-on experience with:

- Building a basic RAG application.
- Loading and preprocessing different document formats.
- Understanding chunk size and overlap.
- Generating and storing text embeddings.
- Using a vector database for semantic retrieval.
- Connecting an LLM to retrieved context.
- Building an interactive AI application with Streamlit.
- Managing API credentials and application configuration.

## 🔮 Future Improvements

- [ ] Display precise source passages and page numbers.
- [ ] Add document deletion and re-indexing.
- [ ] Support multiple documents and collections.
- [ ] Improve retrieval with metadata filtering and reranking.
- [ ] Add conversation-aware question answering.
- [ ] Add document-level access control.
- [ ] Evaluate retrieval quality and answer faithfulness.
- [ ] Deploy the application to a cloud platform.

## ⚠️ Limitations

- Answer quality depends on document extraction, retrieval quality, and the language model.
- Incorrectly extracted text may reduce answer accuracy.
- Source document names do not necessarily identify the exact passage supporting every claim.
- API usage may incur costs.
- The application requires correctly configured API credentials and a compatible Pinecone index.


---

*AskMyDocs is a learning and portfolio project exploring practical applications of Large Language Models and Retrieval-Augmented Generation.*
