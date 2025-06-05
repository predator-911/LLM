import logging
from typing import List, Dict
import httpx
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")
    
    async def generate_response(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate a response using Groq API with retrieved context"""
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Create the prompt
            prompt = self._create_prompt(query, context)
            
            # Call Groq API
            response = await self._call_groq_api(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', 'Unknown')
            content = chunk.get('content', '')
            score = chunk.get('score', 0.0)
            
            context_parts.append(
                f"Source {i} (from {filename}, relevance: {score:.2f}):\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt for the LLM"""
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUERY: {query}

INSTRUCTIONS:
- Answer the query based ONLY on the information provided in the document context above
- If the context doesn't contain enough information to answer the query, clearly state that
- Be specific and cite relevant information from the sources when possible
- Keep your response concise but comprehensive
- If you mention information from the context, indicate which source it came from
- Do not make up information not present in the context

RESPONSE:"""
        
        return prompt
    
    async def _call_groq_api(self, prompt: str) -> str:
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Groq API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                result = response.json()
                
                if 'choices' not in result or len(result['choices']) == 0:
                    error_msg = "Groq API response missing choices"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                # Extract the content from the first choice
                message = result['choices'][0]['message']
                if 'content' not in message:
                    error_msg = "Groq API response missing content in message"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                return message['content'].strip()
                
        except httpx.HTTPError as e:
            error_msg = f"HTTP error occurred: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)