"""
RAG System for TU Wien Informatics
Connects to ChromaDB for retrieval and uses Groq for generation
"""

import os
import tempfile
from typing import List, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, CHROMADB_COLLECTION, LLM_MODEL

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = os.getenv("CHROMADB_PORT")


class TUWienRAG:
    """RAG system for TU Wien informatics programs - connects to existing ChromaDB"""
    
    def __init__(
        self, 
        chroma_host: str = CHROMADB_HOST,
        chroma_port: int = CHROMADB_PORT,
        collection_name: str = CHROMADB_COLLECTION
    ):
        """
        Initialize the RAG system connecting to existing ChromaDB
        
        Args:
            chroma_host: ChromaDB server host
            chroma_port: ChromaDB server port
            collection_name: Name of the collection in ChromaDB
        """
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        self.vector_store = None  # Main store (Docker ChromaDB with pre-scraped docs)
        self.user_vector_store = None  # In-memory store for user-uploaded docs
        self.qa_chain = None
        
        # Initialize embeddings (must match what was used during ingestion)
        print(f"Loading embeddings model: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # Initialize Groq LLM
        print(f"Initializing LLM: {LLM_MODEL}")
        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        # Initialize in-memory ChromaDB for user uploads
        self._init_user_vector_store()
    
    def connect_to_vectorstore(self) -> None:
        """Connect to existing ChromaDB vector store"""
        print(f"Connecting to ChromaDB at {self.chroma_host}:{self.chroma_port}...")
        
        # Connect to ChromaDB server
        client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        
        # Verify collection exists
        collections = [c.name for c in client.list_collections()]
        if self.collection_name not in collections:
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                f"Available collections: {collections}. "
                "Please run document_scraping.py first to ingest documents."
            )
        
        # Connect via LangChain
        self.vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        # Get collection stats
        collection = client.get_collection(self.collection_name)
        doc_count = collection.count()
        print(f"✅ Connected to collection '{self.collection_name}' with {doc_count} documents")
    
    def _init_user_vector_store(self) -> None:
        """Initialize in-memory ChromaDB for user-uploaded documents"""
        print("Initializing in-memory vector store for user uploads...")
        
        # Create ephemeral (in-memory) ChromaDB client
        self.user_chroma_client = chromadb.Client()
        
        # Create collection for user uploads
        self.user_vector_store = Chroma(
            client=self.user_chroma_client,
            collection_name="user_uploads",
            embedding_function=self.embeddings
        )
        print("✅ In-memory vector store ready for user uploads")
    
    def get_user_document_count(self) -> int:
        """Get the number of chunks in the user upload store"""
        try:
            collection = self.user_chroma_client.get_collection("user_uploads")
            return collection.count()
        except Exception:
            return 0
    
    def setup_qa_chain(self) -> None:
        """Set up the QA chain with MMR retrieval for diverse, relevant results"""
        if not self.vector_store:
            raise ValueError("Vector store not connected. Call connect_to_vectorstore() first.")
        
        # Enhanced prompt for TU Wien informatics programs
        prompt_template = """You are a helpful assistant for TU Wien master students in informatics programs.
You have access to information about various master programs, curricula, courses, and admission requirements.

IMPORTANT GUIDELINES:
1. Always answer in English, even if source documents are in German
2. Synthesize information across different sources when relevant
3. Include specific details like course codes, ECTS credits, and prerequisites when available
4. Cite the source (URL or PDF name) for key claims
5. If information is not available in the context, clearly state that

Context from TU Wien documents:
{context}

Question: {question}

Helpful Answer:"""
        

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        print("Setting up QA chain with MMR retrieval...")
        retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 16,
                    "lambda_mult": 0.6
                }
            )
        self.qa_chain = (
                {
                    "context": lambda x: "\n\n".join(ret_doc.page_content for ret_doc in retriever.invoke(x["question"])),
                    "question": lambda x: x["question"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )
        print("✅ QA chain ready")
    
    def _expand_query(self, question: str) -> List[str]:
        """Expand user query to multiple search queries for better retrieval"""
        expansion_prompt = f"""Given this question about TU Wien informatics programs, generate 2 alternative phrasings
that capture different aspects of the question. Return only the 2 queries, one per line, no numbering.

Original question: {question}

Alternative queries:"""
        
        try:
            response = self.llm.invoke(expansion_prompt)
            queries = [question] + [q.strip() for q in response.content.split('\n') if q.strip()]
            return queries[:3]
        except Exception:
            return [question]
    
    def query(self, question: str, conversation_history: List[dict] = None) -> Tuple[str, List]:
        """Query the RAG system with conversation context, searching both stores"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        # Build context from conversation history
        context_messages = ""
        if conversation_history:
            for msg in conversation_history[-4:]:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                context_messages += f"{role}: {content}\n"
        
        # Add conversation context to the question if available
        if context_messages:
            enhanced_question = f"Previous conversation:\n{context_messages}\nNew question: {question}"
        else:
            enhanced_question = question
        
        # Retrieve from both stores and combine with priority
        combined_docs = self._retrieve_from_both_stores(question)
        
        # Build context from combined documents
        context = "\n\n".join([doc.page_content for doc in combined_docs])
        
        # Use LLM directly with combined context
        prompt_template = """You are a helpful assistant for TU Wien master students in informatics programs.
You have access to information about various master programs, curricula, courses, and admission requirements.

IMPORTANT GUIDELINES:
1. Always answer in English, even if source documents are in German
2. Synthesize information across different sources when relevant
3. Include specific details like course codes, ECTS credits, and prerequisites when available
4. Cite the source (URL or PDF name) for key claims
5. If information is not available in the context, clearly state that
6. Pay special attention to information from user-uploaded documents (marked as 'uploaded')

Context from TU Wien documents:
{context}

Question: {question}

Helpful Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        formatted_prompt = prompt.format(context=context, question=enhanced_question)
        response = self.llm.invoke(formatted_prompt)
        
        return response.content, combined_docs
    
    def _retrieve_from_both_stores(
        self, 
        question: str, 
        k_user: int = 3, 
        k_main: int = 4,
        score_threshold: float = 0.3
    ) -> List:
        """
        Retrieve documents from both user uploads (in-memory) and main store (Docker).
        User documents are prioritized and placed first. Low-relevance docs are filtered out.
        
        Args:
            question: The query string
            k_user: Number of documents to retrieve from user store
            k_main: Number of documents to retrieve from main store
            score_threshold: Minimum similarity score (0-1, higher = more similar)
            
        Returns:
            Combined list of relevant documents with user docs first
        """
        combined_docs = []
        
        # First, retrieve from user uploads (in-memory) - these get priority
        user_doc_count = self.get_user_document_count()
        if user_doc_count > 0:
            try:
                effective_k = min(k_user, user_doc_count)
                # Use similarity_search_with_score for relevance filtering
                user_results = self.user_vector_store.similarity_search_with_score(
                    question, 
                    k=effective_k
                )
                # Filter by score and add metadata
                for doc, score in user_results:
                    # ChromaDB returns distance (lower = better), convert to similarity
                    similarity = 1 - score if score <= 1 else 1 / (1 + score)
                    if similarity >= score_threshold:
                        doc.metadata['priority'] = 'high'
                        doc.metadata['source_type'] = 'user_upload'
                        doc.metadata['relevance_score'] = round(similarity, 3)
                        combined_docs.append(doc)
                print(f"📄 Retrieved {len(combined_docs)} relevant chunks from user uploads")
            except Exception as e:
                print(f"Warning: Could not retrieve from user store: {e}")
        
        # Then, retrieve from main store (Docker ChromaDB)
        if self.vector_store:
            try:
                main_results = self.vector_store.similarity_search_with_score(
                    question,
                    k=k_main
                )
                # Filter by score and add metadata
                main_docs_added = 0
                for doc, score in main_results:
                    similarity = 1 - score if score <= 1 else 1 / (1 + score)
                    if similarity >= score_threshold:
                        doc.metadata['priority'] = 'normal'
                        doc.metadata['source_type'] = 'pre_scraped'
                        doc.metadata['relevance_score'] = round(similarity, 3)
                        combined_docs.append(doc)
                        main_docs_added += 1
                print(f"🌐 Retrieved {main_docs_added} relevant chunks from main database")
            except Exception as e:
                print(f"Warning: Could not retrieve from main store: {e}")
        
        return combined_docs
    
    def add_pdf_from_bytes(self, pdf_bytes: bytes, pdf_name: str = "uploaded.pdf") -> int:
        """
        Add a PDF document from bytes to the in-memory vector store.
        
        Args:
            pdf_bytes: The PDF file content as bytes
            pdf_name: Name of the PDF for tracking
            
        Returns:
            Number of chunks added
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            print(f"Loading uploaded PDF: {pdf_name}...")
            loader = PyPDFLoader(tmp_path)
            new_docs = loader.load()
            
            # Add metadata
            for doc in new_docs:
                doc.metadata['type'] = 'pdf'
                doc.metadata['pdf_name'] = pdf_name
                doc.metadata['content_type'] = 'uploaded'
                doc.metadata['source'] = f"Uploaded: {pdf_name}"
            
            print(f"Loaded {len(new_docs)} pages from {pdf_name}")
            
            # Split the new documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            new_docs_split = text_splitter.split_documents(new_docs)
            
            # Add to IN-MEMORY vector store (not the main Docker store)
            if self.user_vector_store:
                print(f"Adding {len(new_docs_split)} chunks to in-memory store...")
                self.user_vector_store.add_documents(new_docs_split)
                print(f"✅ Successfully added {pdf_name} to session storage")
                return len(new_docs_split)
            else:
                raise ValueError("User vector store not initialized")
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def clear_user_documents(self) -> None:
        """Clear all user-uploaded documents from the in-memory store"""
        print("Clearing user-uploaded documents...")
        # Reinitialize the in-memory store
        self._init_user_vector_store()
        print("✅ User documents cleared")
    
    def initialize(self) -> None:
        """Initialize the entire RAG system"""
        self.connect_to_vectorstore()
        self.setup_qa_chain()
        print("🚀 TU Wien RAG system ready!")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the connected collection"""
        client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        collection = client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "document_count": collection.count()
        }


# Backwards compatibility alias
CurriculumRAG = TUWienRAG


# Example usage
if __name__ == "__main__":
    rag = TUWienRAG()
    rag.initialize()
    
    # Test query
    answer, sources = rag.query("What master programs are available in informatics?")
    print("\nQuestion: What master programs are available in informatics?")
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    for doc in sources[:3]:
        source = doc.metadata.get('source', doc.metadata.get('pdf_url', 'Unknown'))
        print(f"- {source}: {doc.page_content[:100]}...")
