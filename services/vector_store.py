import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.vector_store_path = Config.VECTOR_STORE_PATH
        self.embeddings_file = os.path.join(self.vector_store_path, "embeddings.pkl")
        self.metadata_file = os.path.join(self.vector_store_path, "metadata.json")
        
        # Load existing data
        self.embeddings = []
        self.metadata = []
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing embeddings and metadata"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings)} existing embeddings")
            
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} existing metadata entries")
                
        except Exception as e:
            logger.error(f"Error loading existing data: {str(e)}")
            self.embeddings = []
            self.metadata = []
    
    def _save_data(self):
        """Save embeddings and metadata to disk"""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Saved {len(self.embeddings)} embeddings and {len(self.metadata)} metadata entries")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    async def add_documents(self, document_id: str, chunks: List[Dict]):
        """Add document chunks to the vector store"""
        try:
            logger.info(f"Adding {len(chunks)} chunks for document {document_id}")
            
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            chunk_embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Add to storage
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                self.embeddings.append(embedding)
                
                # Create metadata entry
                metadata_entry = {
                    'chunk_id': chunk['id'],
                    'document_id': document_id,
                    'filename': chunk['filename'],
                    'chunk_index': chunk['chunk_index'],
                    'content': chunk['content'],
                    'length': chunk['length'],
                    'created_at': chunk['created_at'],
                    'embedding_index': len(self.embeddings) - 1
                }
                
                self.metadata.append(metadata_entry)
            
            # Save to disk
            self._save_data()
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        try:
            if not self.embeddings:
                logger.warning("No embeddings available for search")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Filter by similarity threshold
            threshold = Config.SIMILARITY_THRESHOLD
            filtered_results = [(i, score) for i, score in top_results if score >= threshold]
            
            # Prepare response
            results = []
            for idx, score in filtered_results:
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['score'] = float(score)
                    results.append(result)
            
            logger.info(f"Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def delete_document(self, document_id: str):
        """Delete all chunks for a document"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for i, metadata in enumerate(self.metadata):
                if metadata['document_id'] == document_id:
                    indices_to_remove.append(i)
            
            if not indices_to_remove:
                logger.warning(f"No chunks found for document {document_id}")
                return
            
            # Remove in reverse order to maintain indices
            indices_to_remove.sort(reverse=True)
            
            for idx in indices_to_remove:
                del self.embeddings[idx]
                del self.metadata[idx]
            
            # Update embedding indices in metadata
            for i, metadata in enumerate(self.metadata):
                metadata['embedding_index'] = i
            
            # Save updated data
            self._save_data()
            
            logger.info(f"Deleted {len(indices_to_remove)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            unique_documents = set()
            total_chunks = len(self.metadata)
            
            for metadata in self.metadata:
                unique_documents.add(metadata['document_id'])
            
            return {
                'total_documents': len(unique_documents),
                'total_chunks': total_chunks,
                'embedding_dimension': Config.EMBEDDING_DIMENSION,
                'model_name': Config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'embedding_dimension': Config.EMBEDDING_DIMENSION,
                'model_name': Config.EMBEDDING_MODEL
            }
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        return [
            metadata for metadata in self.metadata
            if metadata['document_id'] == document_id
        ]
