from typing import List, Dict, Optional
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from .document_processor import DocumentProcessor

class ChatService:
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.model = ChatAnthropic(
            model=os.getenv("DEFAULT_MODEL", "claude-3-haiku-20240307"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the QA chain with the appropriate prompt template."""
        prompt_template = """You are an expert financial analyst assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Please provide a detailed analysis based on the available information. If analyzing financial metrics or trends,
        break down the key points clearly. For risk factors or market analysis, highlight the most important aspects
        and their potential impact.
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.document_processor.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

    async def process_message(self, messages: List[Dict], context_ids: Optional[List[str]] = None) -> Dict:
        """Process a chat message and return the response."""
        # Extract the latest user message
        user_message = messages[-1]["content"]
        
        try:
            # If specific documents are referenced, filter the search
            if context_ids:
                # TODO: Implement filtered search based on document IDs
                pass
            
            # Get the response from the QA chain
            response = self.qa_chain({"question": user_message})
            
            # Extract source documents for citation
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response.get("source_documents", [])
            ]
            
            return {
                "response": response["answer"],
                "sources": sources
            }
            
        except Exception as e:
            raise Exception(f"Error processing message: {str(e)}")

    def reset_conversation(self):
        """Reset the conversation memory."""
        self.memory.clear() 