import os
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.database import DatabaseService
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

class DocumentMetadata(BaseModel):
    filename: str
    file_size: int
    pages: int
    chunks: int
    upload_date: str
    document_id: str

# Global services
document_processor = None
vector_store = None
llm_service = None
db_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global document_processor, vector_store, llm_service, db_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize services
        document_processor = DocumentProcessor()
        vector_store = VectorStore()
        llm_service = LLMService()
        db_service = DatabaseService()
        
        # Initialize database
        await db_service.initialize()
        
        logger.info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down services...")

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="A Retrieval-Augmented Generation pipeline for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": "operational"}

@app.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (max 50MB)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        # Check file type
        allowed_types = ['.pdf', '.txt', '.docx', '.md']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_types):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        logger.info(f"Processing file: {file.filename}")
        
        # Process document
        document_id, chunks = await document_processor.process_document(
            content, file.filename
        )
        
        # Store in vector database
        await vector_store.add_documents(document_id, chunks)
        
        # Store metadata in database
        metadata = {
            'document_id': document_id,
            'filename': file.filename,
            'file_size': len(content),
            'chunks': len(chunks),
            'pages': await document_processor.get_page_count(content, file.filename)
        }
        
        await db_service.store_document_metadata(metadata)
        
        logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks")
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Retrieve relevant chunks
        relevant_chunks = await vector_store.search(request.query, request.top_k)
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the uploaded documents.",
                sources=[],
                query=request.query
            )
        
        # Generate response using LLM
        answer = await llm_service.generate_response(request.query, relevant_chunks)
        
        # Format sources
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "page": chunk.get("page", "Unknown"),
                "similarity_score": chunk.get("score", 0.0),
                "preview": chunk.get("content", "")[:200] + "..."
            })
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentMetadata])
async def get_documents():
    """Get all uploaded document metadata"""
    try:
        documents = await db_service.get_all_documents()
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        # Remove from vector store
        await vector_store.delete_document(document_id)
        
        # Remove from database
        await db_service.delete_document(document_id)
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = await db_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )