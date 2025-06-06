# RAG Pipeline - Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) pipeline that allows users to upload documents and ask questions based on their content. Built with FastAPI, using Groq for LLM inference and free embedding models.

## 🚀 Features

- **Document Upload & Processing**: Support for PDF, DOCX, TXT, and Markdown files
- **Intelligent Chunking**: Smart text splitting with configurable overlap
- **Vector Search**: Efficient similarity search using sentence transformers
- **LLM Integration**: Powered by Groq API for fast, accurate responses
- **REST API**: Complete FastAPI implementation with automatic docs
- **Free Deployment**: Optimized for free tier hosting on Render
- **Persistent Storage**: SQLite database for metadata, file-based vector storage
- **Docker Support**: Complete containerization for any environment

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **LLM**: Groq API (Llama 3)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Storage**: Custom implementation with pickle serialization
- **Database**: SQLite (for free deployment)
- **Document Processing**: PyPDF2, python-docx
- **Deployment**: Docker, Render

## 📋 Prerequisites

- Python 3.11 or higher
- Groq API key (free at https://console.groq.com)
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-pipeline
```

### 2. Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
EMBEDDING_MODEL=all-MiniLM-L6-v2
ENVIRONMENT=development
DEBUG=true
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

## 🐳 Docker Deployment

### Local Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t rag-pipeline .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here rag-pipeline
```

## ☁️ Render Deployment (Free Tier)

### Step 1: Prepare Your Repository

1. Push your code to GitHub
2. Ensure `render.yaml` is in your repository root

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect the `render.yaml` configuration
5. Add your `GROQ_API_KEY` in the environment variables
6. Click "Apply"

### Step 3: Configure Environment Variables

In your Render service settings, add:
- `GROQ_API_KEY`: Your Groq API key

## 📚 API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 API Endpoints

### Document Management

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

# Upload a file
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -F "file=@document.pdf"
```

#### Get All Documents
```http
GET /documents

curl -X GET "http://localhost:8000/documents"
```

#### Delete Document
```http
DELETE /documents/{document_id}

curl -X DELETE "http://localhost:8000/documents/abc-123"
```

### Query System

#### Ask Questions
```http
POST /query
Content-Type: application/json

{
  "query": "What is the main topic of the documents?",
  "top_k": 5
}
```

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "top_k": 3}'
```

### System Information

#### Health Check
```http
GET /health
```

#### System Statistics
```http
GET /stats
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Required**: Your Groq API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Default number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score |
| `MAX_FILE_SIZE_MB` | `50` | Maximum file size |
| `DATABASE_URL` | `sqlite:///./data/documents.db` | Database connection string |

### File Structure

```
rag-pipeline/
├── app.py                 # Main FastAPI application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── render.yaml          # Render deployment config
├── services/
│   ├── document_processor.py  # Document processing logic
│   ├── vector_store.py        # Vector storage and search
│   ├── llm_service.py         # Groq API integration
│   └── database.py            # SQLite database operations
├── tests/
│   └── test_api.py           # API tests
├── data/                     # Data storage (created automatically)
│   ├── vector_store/         # Vector embeddings
│   └── documents.db          # SQLite database
└── README.md
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Manual Testing

1. **Upload a document**:
```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@sample_document.pdf"
```

2. **Query the system**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic?"}'
```

## 📊 Performance Optimization

### For Production

1. **Increase chunk size** for longer documents:
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

2. **Adjust similarity threshold**:
```env
SIMILARITY_THRESHOLD=0.6
```

3. **Use better embedding models** (if resources allow):
```env
EMBEDDING_MODEL=all-mpnet-base-v2
```

## 🔒 Security Considerations

1. **API Key Security**: Never commit API keys to version control
2. **File Upload Limits**: Configured maximum file sizes
3. **Input Validation**: All inputs are validated using Pydantic
4. **CORS**: Configure appropriately for your domain
