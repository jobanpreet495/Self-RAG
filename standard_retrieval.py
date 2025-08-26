from typing import List, Dict

import logging
import hashlib
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 


load_dotenv()

logger = logging.getLogger(__name__)

class StandardRetriever:
    """Complete RAG system with Self-RAG integration and persistent storage"""
    
    def __init__(self, urls: List[str],
                 persist_dir: str = "./vectorstore"):
       
        self.persist_dir = persist_dir
        
        # Create unique collection name based on URLs
        urls_hash = hashlib.md5(str(sorted(urls)).encode()).hexdigest()[:8]
        self.collection_name = f"selfrag-{urls_hash}"
        
        self._setup_retrieval_system(urls)
        
    def _setup_retrieval_system(self, urls: List[str]):
        """Set up document indexing and retrieval with persistence"""
        try:
            # Try to load existing vectorstore first
            if os.path.exists(self.persist_dir):
                try:
                    self.vectorstore = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=OpenAIEmbeddings(),
                        persist_directory=self.persist_dir
                    )
                    
                    # Check if collection has documents
                    if self.vectorstore._collection.count() > 0:
                        self.retriever = self.vectorstore.as_retriever(
                            search_kwargs={"k": 5}
                        )
                        logger.info(f"Loaded existing vectorstore with {self.vectorstore._collection.count()} documents")
                        
                        return
                except Exception as e:
                    logger.warning(f"Could not load existing vectorstore: {e}")
            
            # Create new vectorstore if loading failed or doesn't exist
            logger.info("Creating new vectorstore...")
            
            # Load documents
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]
          
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            doc_splits = text_splitter.split_documents(docs_list)
            
            # Create vector store with persistence
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=OpenAIEmbeddings(),
                persist_directory=self.persist_dir
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            logger.info(f"Created and saved vectorstore with {len(doc_splits)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to setup retrieval system: {e}")
            self.retriever = None
    
    def retrieve(self, question: str) -> Dict:
        """Main query interface that combines retrieval with CRAG"""
        if not self.retriever:
            return {"error": "Retrieval system not available"}
        
        # Retrieve documents
        retrieved_docs = self.retriever.invoke(question)
        
    
        return retrieved_docs
          
  


