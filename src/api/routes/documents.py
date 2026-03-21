"""
Document management API routes.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from typing import List, Dict, Any
import uuid
import logging
from pathlib import Path
import tempfile
import os

from ...core.database import VectorDatabase
from ...utils.preprocessing import DocumentProcessor
from ...models.schemas import UploadResponse, StatusResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Dependency to get database instance
def get_database():
    return VectorDatabase()

# Dependency to get document processor
def get_document_processor():
    return DocumentProcessor()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: VectorDatabase = Depends(get_database),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """Upload and process a document for retrieval."""
    try:
        # Validate file type
        allowed_types = {".pdf", ".txt", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed types: {allowed_types}"
            )
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document
            chunks = processor.process_file(temp_file_path, file.filename)
            
            # Store in vector database
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                db.add_document(
                    doc_id=chunk_id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "document_id": doc_id,
                        "filename": file.filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
            
            logger.info(f"Successfully uploaded document {file.filename} with {len(chunks)} chunks")
            
            return UploadResponse(
                success=True,
                document_id=doc_id,
                filename=file.filename,
                message=f"Document uploaded successfully. Created {len(chunks)} chunks.",
                processing_time=0.0  # You could add timing here
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: VectorDatabase = Depends(get_database)
):
    """Delete a document and all its chunks."""
    try:
        # Delete all chunks for this document
        deleted_count = db.delete_by_metadata({"document_id": document_id})
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
            
        logger.info(f"Deleted document {document_id} ({deleted_count} chunks)")
        
        return StatusResponse(
            status="success",
            message=f"Document {document_id} deleted successfully ({deleted_count} chunks removed)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/documents")
async def list_documents(
    db: VectorDatabase = Depends(get_database)
) -> Dict[str, Any]:
    """List all documents in the database."""
    try:
        # Get all unique documents
        documents = db.get_all_documents()
        
        # Group by document_id
        doc_map = {}
        for doc in documents:
            doc_id = doc.get("metadata", {}).get("document_id", "unknown")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": doc.get("metadata", {}).get("filename", "unknown"),
                    "chunks": 0,
                    "total_chunks": doc.get("metadata", {}).get("total_chunks", 0)
                }
            doc_map[doc_id]["chunks"] += 1
        
        return {
            "documents": list(doc_map.values()),
            "total_documents": len(doc_map)
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
