import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add the parent directory to sys.path to import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config import Config

client = TestClient(app)

@pytest.fixture
def sample_pdf_content():
    """Create a simple PDF content for testing"""
    # This is a minimal PDF content - in real tests you'd use a proper PDF
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

@pytest.fixture
def sample_text_content():
    """Create sample text content"""
    return "This is a sample document for testing. It contains multiple sentences for chunking."

class TestHealthEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "RAG Pipeline API is running!" in response.json()["message"]
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestDocumentUpload:
    def test_upload_text_file(self, sample_text_content):
        """Test uploading a text file"""
        files = {"file": ("test.txt", sample_text_content, "text/plain")}
        
        with patch('services.document_processor.DocumentProcessor.process_document') as mock_process:
            mock_process.return_value = ("test-id", [{"id": "chunk1", "content": "test"}])
            
            with patch('services.vector_store.VectorStore.add_documents') as mock_vector:
                with patch('services.database.DatabaseService.store_document_metadata') as mock_db:
                    response = client.post("/upload", files=files)
                    
                    assert response.status_code == 200
                    assert "Document uploaded and processed successfully" in response.json()["message"]
    
    def test_upload_unsupported_file(self):
        """Test uploading an unsupported file type"""
        files = {"file": ("test.exe", b"binary content", "application/octet-stream")}
        
        response = client.post("/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_no_file(self):
        """Test uploading without a file"""
        response = client.post("/upload")
        assert response.status_code == 422  # Validation error

class TestQuerying:
    def test_query_with_results(self):
        """Test querying with mock results"""
        query_data = {"query": "What is the main topic?", "top_k": 3}
        
        mock_chunks = [
            {
                "id": "chunk1",
                "content": "This is about machine learning",
                "document_id": "doc1",
                "filename": "test.txt",
                "score": 0.85
            }
        ]
        
        with patch('services.vector_store.VectorStore.search') as mock_search:
            mock_search.return_value = mock_chunks
            
            with patch('services.llm_service.LLMService.generate_response') as mock_llm:
                mock_llm.return_value = "This document is about machine learning."
                
                response = client.post("/query", json=query_data)
                
                assert response.status_code == 200
                result = response.json()
                assert "answer" in result
                assert "sources" in result
                assert len(result["sources"]) > 0
    
    def test_query_no_results(self):
        """Test querying with no results"""
        query_data = {"query": "What is the main topic?"}
        
        with patch('services.vector_store.VectorStore.search') as mock_search:
            mock_search.return_value = []
            
            response = client.post("/query", json=query_data)
            
            assert response.status_code == 200
            result = response.json()
            assert "couldn't find any relevant information" in result["answer"]
    
    def test_query_invalid_data(self):
        """Test querying with invalid data"""
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error

class TestDocumentManagement:
    def test_get_documents(self):
        """Test getting all documents"""
        mock_docs = [
            {
                "document_id": "doc1",
                "filename": "test.txt",
                "file_size": 1000,
                "pages": 1,
                "chunks": 5,
                "upload_date": "2024-01-01T00:00:00"
            }
        ]
        
        with patch('services.database.DatabaseService.get_all_documents') as mock_db:
            mock_db.return_value = mock_docs
            
            response = client.get("/documents")
            
            assert response.status_code == 200
            assert len(response.json()) == 1
            assert response.json()[0]["filename"] == "test.txt"
    
    def test_delete_document(self):
        """Test deleting a document"""
        with patch('services.vector_store.VectorStore.delete_document') as mock_vector:
            with patch('services.database.DatabaseService.delete_document') as mock_db:
                response = client.delete("/documents/test-doc-id")
                
                assert response.status_code == 200
                assert "deleted successfully" in response.json()["message"]
    
    def test_get_stats(self):
        """Test getting system statistics"""
        mock_stats = {
            "documents": {"total": 5},
            "queries_last_30_days": {"total": 100}
        }
        
        with patch('services.database.DatabaseService.get_stats') as mock_db:
            mock_db.return_value = mock_stats
            
            response = client.get("/stats")
            
            assert response.status_code == 200
            assert response.json()["documents"]["total"] == 5

class TestConfiguration:
    def test_config_validation(self):
        """Test configuration validation"""
        # Test that config validation works
        original_key = os.environ.get('GROQ_API_KEY', '')
        
        # Test with missing API key
        if 'GROQ_API_KEY' in os.environ:
            del os.environ['GROQ_API_KEY']
        
        with pytest.raises(ValueError):
            Config.validate_config()
        
        # Restore original key
        if original_key:
            os.environ['GROQ_API_KEY'] = original_key

# Integration tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Test the full pipeline with mocked services"""
        # This would test the entire flow from upload to query
        # In a real scenario, you'd use actual test files and test the full flow
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])