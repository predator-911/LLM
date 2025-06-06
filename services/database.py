import aiosqlite
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
import os

from config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.db_path = Config.DATABASE_URL.replace("sqlite:///", "")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    async def initialize(self):
        """Initialize the database with required tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create documents table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        pages INTEGER NOT NULL,
                        chunks INTEGER NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create queries table for analytics
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        response_time REAL,
                        chunks_retrieved INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    async def store_document_metadata(self, metadata: Dict):
        """Store document metadata"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO documents (id, filename, file_size, pages, chunks, upload_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metadata['document_id'],
                    metadata['filename'],
                    metadata['file_size'],
                    metadata['pages'],
                    metadata['chunks'],
                    datetime.utcnow().isoformat()
                ))
                
                await db.commit()
                logger.info(f"Stored metadata for document: {metadata['filename']}")
                
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            raise
    
    async def get_all_documents(self) -> List[Dict]:
        """Get all document metadata"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT id, filename, file_size, pages, chunks, upload_date
                    FROM documents
                    ORDER BY upload_date DESC
                """)
                
                rows = await cursor.fetchall()
                
                documents = []
                for row in rows:
                    documents.append({
                        'document_id': row[0],
                        'filename': row[1],
                        'file_size': row[2],
                        'pages': row[3],
                        'chunks': row[4],
                        'upload_date': row[5]
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """Get document metadata by ID"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT id, filename, file_size, pages, chunks, upload_date
                    FROM documents
                    WHERE id = ?
                """, (document_id,))
                
                row = await cursor.fetchone()
                
                if row:
                    return {
                        'document_id': row[0],
                        'filename': row[1],
                        'file_size': row[2],
                        'pages': row[3],
                        'chunks': row[4],
                        'upload_date': row[5]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            return None
    
    async def delete_document(self, document_id: str):
        """Delete document metadata"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    DELETE FROM documents WHERE id = ?
                """, (document_id,))
                
                await db.commit()
                logger.info(f"Deleted document metadata: {document_id}")
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    async def log_query(self, query_text: str, response_time: float, chunks_retrieved: int):
        """Log query for analytics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO queries (query_text, response_time, chunks_retrieved)
                    VALUES (?, ?, ?)
                """, (query_text, response_time, chunks_retrieved))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            # Don't raise here as it's not critical
    
    async def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get document stats
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(file_size) as total_size,
                        SUM(pages) as total_pages,
                        SUM(chunks) as total_chunks,
                        AVG(file_size) as avg_file_size
                    FROM documents
                """)
                doc_stats = await cursor.fetchone()
                
                # Get query stats
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(response_time) as avg_response_time,
                        AVG(chunks_retrieved) as avg_chunks_retrieved
                    FROM queries
                    WHERE created_at >= datetime('now', '-30 days')
                """)
                query_stats = await cursor.fetchone()
                
                return {
                    'documents': {
                        'total': doc_stats[0] or 0,
                        'total_size_bytes': doc_stats[1] or 0,
                        'total_pages': doc_stats[2] or 0,
                        'total_chunks': doc_stats[3] or 0,
                        'average_file_size_bytes': doc_stats[4] or 0
                    },
                    'queries_last_30_days': {
                        'total': query_stats[0] or 0,
                        'average_response_time_seconds': query_stats[1] or 0,
                        'average_chunks_retrieved': query_stats[2] or 0
                    },
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'documents': {
                    'total': 0,
                    'total_size_bytes': 0,
                    'total_pages': 0,
                    'total_chunks': 0,
                    'average_file_size_bytes': 0
                },
                'queries_last_30_days': {
                    'total': 0,
                    'average_response_time_seconds': 0,
                    'average_chunks_retrieved': 0
                },
                'generated_at': datetime.utcnow().isoformat()
            }