# Research Paper RAG Assistant

A complete, runnable **Retrieval-Augmented Generation (RAG)** application for academic research papers. Upload PDFs, ask natural-language questions, and get concise, synthesized answers with **inline citations** (source paper + page number).

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Frontend   │     │                   Backend                    │
│  React+Vite │     │                 FastAPI (Py3.11)             │
│  (SSE client)│────▶│                                              │
└─────────────┘     │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
                    │  │  PDF     │  │  ColPali │  │  ChromaDB  │  │
                    │  │  Parser  │──▶│ Retriever│──▶│ (vector DB)│  │
                    │  └──────────┘  └──────────┘  └────────────┘  │
                    │        │              │                       │
                    │        ▼              ▼                       │
                    │  ┌──────────┐  ┌──────────────┐              │
                    │  │  Groq    │◀─│  Retrieved   │              │
                    │  │  LLM     │  │  context     │              │
                    │  └──────────┘  └──────────────┘              │
                    └──────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Role |
|-----------|-----------|------|
| **PDF Parsing** | `pypdf` + `pypdfium2` | Extract per-page text + render each page to an image |
| **Embedding / Retrieval** | **ColPali** (`vidore/colpali-v1.2`) | Multimodal late-interaction retrieval — treats pages as images, capturing tables, figures, and complex layouts |
| **Vector DB** | **ChromaDB** (persistent, local) | Stores page embeddings + metadata (filename, page number) |
| **LLM** | **Groq API** + Llama 3.3 70B | Generates synthesized answers with inline citations |
| **Backend** | **FastAPI** (Python 3.11+) | Ingestion pipeline, retrieval, SSE streaming |
| **Frontend** | **React + Vite** | Drag-and-drop upload, chat UI with conversation history |

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # pydantic-settings config (env vars)
│   │   ├── main.py            # FastAPI app, /upload, /ask (SSE), /health
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── pdf_parser.py      # pypdf + pypdfium2 page text/image extraction
│   │   ├── retriever.py       # ColPali embeddings + ChromaDB retrieval
│   │   └── llm.py             # Groq streaming LLM wrapper
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx            # Upload + chat UI with SSE streaming
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
└── README.md
```

## Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend dev)
- **Docker + Docker Compose** (optional, for containerized run)
- **Groq API key** — get one at [console.groq.com](https://console.groq.com)

### Option A: Docker Compose (recommended)

```bash
# 1. Set your Groq API key
export GROQ_API_KEY=your_key_here

# 2. Build and start
docker compose up --build

# 3. Open the UI
#    Frontend:  http://localhost:8080
#    Backend:   http://localhost:8000/docs
```

### Option B: Local development

**Backend:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set env vars (or copy .env.example to .env and edit)
export GROQ_API_KEY=your_key_here

uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `COLPALI_MODEL_NAME` | `vidore/colpali-v1.2` | ColPali model for embeddings |
| `COLPALI_DEVICE` | `cpu` | `cpu` or `cuda` (GPU) |
| `TOP_K` | `5` | Number of pages to retrieve per query |
| `HOST` | `0.0.0.0` | Backend bind host |
| `PORT` | `8000` | Backend bind port |

## API Reference

### `POST /upload`

Upload and ingest a PDF (multipart form-data, field name `file`).

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@paper.pdf"
```

**Response (200):**
```json
{
  "filename": "paper.pdf",
  "pages": 12,
  "status": "ingested",
  "message": "Ingested 12 pages from paper.pdf."
}
```

### `POST /ask`

Ask a question. Returns a **Server-Sent Events (SSE)** stream.

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?", "top_k": 5}'
```

**SSE Response:**
```
data: {"type": "sources", "sources": [{"filename": "paper.pdf", "page": 3, "distance": 0.42}, ...]}

data: {"type": "token", "content": "The main contribution is a novel ..."}

data: {"type": "token", "content": " attention mechanism [Author et al., 2021, p. 3]."}

data: {"type": "done"}
```

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "documents": 12,
  "model": "vidore/colpali-v1.2"
}
```

## How It Works

### 1. Ingestion Pipeline

1. **Upload** — PDF received via `POST /upload`.
2. **Parse** — `pypdf` extracts per-page text; `pypdfium2` renders each page to an image (144 DPI).
3. **Embed** — ColPali processes each page image into a late-interaction embedding (mean-pooled to a single vector for ChromaDB).
4. **Store** — Each page is stored in ChromaDB with metadata: `filename`, `page`, `total_pages`.

### 2. Query / Chat

1. **Retrieve** — User question is embedded with ColPali; top-K most relevant pages are fetched from ChromaDB (cosine similarity).
2. **Context** — Retrieved pages are formatted into a context block with source metadata.
3. **Generate** — Groq Llama 3.3 streams a synthesized answer, instructed to cite sources inline as `[Author et al., YEAR, p. N]`.
4. **Stream** — The answer streams back to the frontend via SSE, along with the retrieved source list.

### 3. Source Attribution

The system prompt enforces citation rules:
- Every factual claim must be followed by `[Author et al., YEAR, p. N]`.
- The frontend also displays the retrieved source pages with relevance scores.

## Example

**Question:** *"What method does the paper propose for handling long documents?"*

**Answer (streamed):**
> The paper proposes a **hierarchical attention mechanism** that processes documents in two stages: first, it segments the text into semantic blocks, then applies cross-block attention to capture long-range dependencies [Author et al., 2023, p. 5]. This reduces memory usage by 40% compared to full attention [Author et al., 2023, p. 7].

**Sources panel:**
```
paper.pdf — p. 5 (score: 0.412)
paper.pdf — p. 7 (score: 0.388)
```

## Notes

- **First run** downloads the ColPali model (~1.5 GB) and may take a few minutes.
- **GPU**: Set `COLPALI_DEVICE=cuda` for faster embedding. Requires a CUDA-enabled PyTorch install.
- **ChromaDB** persists to `./data/chroma` — delete this directory to reset the index.
- The `pdf2image` package is listed in requirements but `pypdfium2` is used for rendering (no poppler dependency). `pdf2image` is kept as an alternative if you prefer poppler-based rendering.

## License

MIT