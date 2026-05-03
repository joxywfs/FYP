# FYP ChatGPT Retrieval Plugin Development
A lightweight Retrieval-Augmented Generation (RAG) demo built with **FastAPI** and a simple frontend interface.  
This project allows users to upload documents, index them into a vector database, and perform semantic search over the stored document chunks.

The demo frontend is designed to provide a more user-friendly interface than Swagger UI. Users can upload documents, run searches, and inspect retrieved chunks directly from the browser.

---

## Features
- Upload and index documents into the knowledge base
- Perform semantic search on uploaded documents
- Display search results with:
  - source file name
  - chunk index
  - word count
  - chunk ID
  - similarity score
  - metadata details
- View backend health status
- View database statistics such as:
  - total number of documents
  - total number of chunks
  - last updated timestamp
- Simple browser-based demo page for testing the retrieval system

---

## Supported File Formats
The demo currently supports:
- TXT  
- PDF  
- DOCX  
- HTML  
- MD  

---

## Project Setup
This project was run using **VS Code**.
Please have .env ready with API key inside for this code to run properly.

### 1. Create and activate a virtual environment
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Application
Run the backend server:
```bash
uvicorn src.api.main:app --reload
```

## API / Demo UI
Open the application in your browser: \
`http://127.0.0.1:8000`

The demo page will automatically load.

From the demo page, you can:
1. Upload a document into the knowledge base
2. Refresh and view system statistics
3. Enter a search query
4. Configure:
   - maximum number of results
   - score threshold
5. Run semantic search and inspect returned document chunks

## How the Demo Works
### Upload Document
* Upload files via the UI
* Backend endpoint: \
`/api/v1/documents/upload`

After upload:
* document is processed
* text is chunked
* embeddings are generated
* chunks are stored in the vector database

### Search Knowledge Base
* Enter a query in the UI
* Backend endpoint: \
`/api/v1/search`

Parameters:
* query
* max_results
* score_threshold
* rerank = false

### Health Check
`/api/v1/health` \
Indicates whether the API is running.

### System Stats
`/api/v1/stats`\
\
Displays:
* total documents
* total chunks
* last updated time
* API base path

## Reset / Clear Database
To clear all indexed data:
```bash
Remove-Item -Recurse -Force chroma_db
```

## Notes
* This demo is for demonstration purposes only
* Documents must be uploaded before searching
* If API is not running, the frontend will show an error
* If no results:
  * lower score threshold
  * use a more specific query
  * upload more documents

## Example Workflow
1. Activate virtual environment
2. Run FastAPI server
3. Open `http://127.0.0.1:8000`
4. Upload document
5. Wait for indexing
6. Enter query (e.g., machine learning)
7. Adjust parameters
8. View results

## Tech Stack
* Backend: FastAPI
* Frontend: HTML, CSS, JavaScript
* Vector Database: ChromaDB
* Embedding & Retrieval: Custom pipeline

## Disclaimer
This project demonstrates a ChatGPT-style retrieval system using document indexing and semantic search.
