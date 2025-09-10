from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path

from .services.document_processor import DocumentProcessor
from .services.chat_service import ChatService
from .services.sentiment_analyzer import SentimentAnalyzer

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Analyst Assistant",
    description="AI-powered financial analysis and document processing API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor(
    store_dir=os.getenv("DOCUMENT_STORE_DIR", "./data/documents"),
    persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

chat_service = ChatService(document_processor)
sentiment_analyzer = SentimentAnalyzer(
    model_name=os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
)

# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context_ids: Optional[List[str]] = None

class DocumentMetadata(BaseModel):
    title: str
    source: str
    date: Optional[str]
    document_type: str
    company: Optional[str]

class UploadResponse(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    status: str

class SentimentRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = await chat_service.process_message(
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            context_ids=request.context_ids
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    company: Optional[str] = None
):
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "source": "upload",
            "date": None,
            "document_type": "pdf",
            "company": company
        }
        
        # Process document in background
        doc_id = document_processor.process_pdf(tmp_path, metadata)
        
        # Clean up temporary file in background
        background_tasks.add_task(os.unlink, tmp_path)
        
        return {
            "document_id": doc_id,
            "metadata": metadata,
            "status": "processed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = sentiment_analyzer.analyze_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    try:
        chunks = document_processor.get_document_chunks(doc_id)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    ) 