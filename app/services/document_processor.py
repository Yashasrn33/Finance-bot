from typing import Dict, List, Optional
import os
from pathlib import Path
import hashlib
import json
from datetime import datetime

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self, store_dir: str = "./data/documents", 
                 persist_dir: str = "./data/chroma",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.store_dir = Path(store_dir)
        self.persist_dir = Path(persist_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path: str, metadata: Dict) -> str:
        """Process a PDF file and store its chunks in the vector store."""
        # Generate document ID
        doc_id = self._generate_doc_id(file_path)
        
        try:
            # Load and parse PDF
            loader = UnstructuredPDFLoader(file_path)
            document = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(document)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update({
                    'document_id': doc_id,
                    'chunk_id': hashlib.md5(chunk.page_content.encode()).hexdigest(),
                    'processed_at': datetime.now().isoformat(),
                    **metadata
                })
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save metadata
            self._save_metadata(doc_id, metadata)
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file content and timestamp."""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        return f"{content_hash}_{timestamp}"

    def _save_metadata(self, doc_id: str, metadata: Dict):
        """Save document metadata to disk."""
        metadata_path = self.store_dir / f"{doc_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks in the vector store."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in results
        ]

    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Retrieve all chunks for a specific document."""
        results = self.vector_store.get(
            where={'document_id': doc_id}
        )
        return results if results else [] 