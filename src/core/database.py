import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any, Tuple
import uuid
import logging
from pathlib import Path
import json
from datetime import datetime

from ..core.config import settings
from ..utils.preprocessing import DocumentChunk

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database implementation using ChromaDB."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.collection_name
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with new API
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        # Track statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_updated": None
        }
        self._update_stats()
    
    def _get_or_create_collection(self):
        """Initialize or retrieve the main collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> Tuple[bool, str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents with 'id', 'embedding', 'content', and 'metadata'
            batch_size: Number of documents to process in each batch
            
        Returns:
            Tuple of (success, message)
        """
        try:
            total_docs = len(documents)
            added_count = 0
            
            # Process in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                ids = []
                embeddings = []
                metadatas = []
                documents_text = []
                
                for doc in batch:
                    # Validate document structure
                    if not all(key in doc for key in ['id', 'embedding', 'content', 'metadata']):
                        logger.warning(f"Skipping invalid document: {doc.get('id', 'unknown')}")
                        continue
                    
                    ids.append(doc['id'])
                    embeddings.append(doc['embedding'])
                    metadatas.append(doc['metadata'])
                    documents_text.append(doc['content'])
                
                if ids:  # Only add if we have valid documents
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents_text
                    )
                    added_count += len(ids)
                    
                    logger.info(
                        f"Added batch {i//batch_size + 1}: "
                        f"{len(ids)} documents (Total: {added_count}/{total_docs})"
                    )
            
            # Update statistics
            self._update_stats()
            
            # Persist changes
            #self.client.persist()
            
            return True, f"Successfully added {added_count} documents"
            
        except Exception as e:
            error_msg = f"Error adding documents: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> Tuple[bool, str]:
        """Add document chunks with their embeddings."""
        if len(chunks) != len(embeddings):
            return False, "Number of chunks and embeddings must match"
        
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                'id': chunk.chunk_id,
                'embedding': embedding,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            documents.append(doc)
        
        return self.add_documents(documents)
    
    def search(
        self, 
        query_embedding: List[float], 
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        include_distance: bool = True
    ) -> Dict[str, Any]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_dict: Optional metadata filters
            include_distance: Whether to include distance scores
            
        Returns:
            Dictionary with search results
        """
        try:
            # Build where clause for filtering
            where_clause = filter_dict if filter_dict else None
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                'ids': results['ids'][0] if results['ids'] else [],
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if include_distance and results['distances'] else [],
                'scores': []  # Will calculate similarity scores
            }
            
            # Convert distances to similarity scores (1 - distance for cosine)
            if include_distance and formatted_results['distances']:
                formatted_results['scores'] = [
                    1 - dist for dist in formatted_results['distances']
                ]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': [],
                'scores': []
            }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['embeddings', 'documents', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0] if results['embeddings'] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def update_document(
        self, 
        doc_id: str, 
        embedding: Optional[List[float]] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Update an existing document."""
        try:
            # Get existing document
            existing = self.get_document_by_id(doc_id)
            if not existing:
                return False, f"Document {doc_id} not found"
            
            # Update fields
            if embedding is not None:
                self.collection.update(
                    ids=[doc_id],
                    embeddings=[embedding]
                )
            
            if content is not None or metadata is not None:
                update_data = {}
                if content is not None:
                    update_data['documents'] = [content]
                if metadata is not None:
                    update_data['metadatas'] = [metadata]
                
                self.collection.update(ids=[doc_id], **update_data)
            
            #self.client.persist()
            return True, f"Successfully updated document {doc_id}"
            
        except Exception as e:
            error_msg = f"Error updating document: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_document(self, doc_id: str) -> Tuple[bool, str]:
        """Delete a document from the database."""
        try:
            self.collection.delete(ids=[doc_id])
            #self.client.persist()
            self._update_stats()
            return True, f"Successfully deleted document {doc_id}"
            
        except Exception as e:
            error_msg = f"Error deleting document: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_by_source(self, source_path: str) -> Tuple[bool, str]:
        """Delete all documents from a specific source."""
        try:
            # Query all documents from this source
            results = self.collection.get(
                where={"source_path": source_path}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
                #self.client.persist()
                self._update_stats()
                return True, f"Deleted {len(results['ids'])} documents from {source_path}"
            
            return True, "No documents found from this source"
            
        except Exception as e:
            error_msg = f"Error deleting by source: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def clear_collection(self) -> Tuple[bool, str]:
        """Clear all documents from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate it
            self.collection = self._get_or_create_collection()
            self._update_stats()
            
            return True, "Collection cleared successfully"
            
        except Exception as e:
            error_msg = f"Error clearing collection: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        self._update_stats()
        return self.stats
    
    def _update_stats(self):
        """Update internal statistics."""
        try:
            count = self.collection.count()
            self.stats.update({
                "total_chunks": count,
                "last_updated": datetime.now().isoformat()
            })
            
            # Count unique sources
            results = self.collection.get(include=['metadatas'])
            if results['metadatas']:
                sources = set()
                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
                self.stats['total_documents'] = len(sources)
                
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def export_collection(self, export_path: str) -> Tuple[bool, str]:
        """Export collection data to a file."""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all data
            results = self.collection.get(
                include=['embeddings', 'documents', 'metadatas']
            )
            
            # Prepare export data
            export_data = {
                'collection_name': self.collection_name,
                'stats': self.get_statistics(),
                'documents': []
            }
            
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': results['embeddings'][i] if results['embeddings'] else None
                }
                export_data['documents'].append(doc)
            
            # Save to file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True, f"Exported {len(export_data['documents'])} documents to {export_path}"
            
        except Exception as e:
            error_msg = f"Error exporting collection: {str(e)}"
            logger.error(error_msg)
            return False, error_msg