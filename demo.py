import os
import json
import glob
import warnings
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- 1. CONFIGURATION & SETUP ---
warnings.filterwarnings("ignore")
os.environ["GROQ_API_KEY"] = "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"

DATA_FOLDER = "./data"
DB_PERSIST_DIR = "./chroma_db_storage" 

console = Console()

console.print(Panel("[bold blue]TULens AI System Initializing...[/bold blue]"))

# --- 2. CORE AI COMPONENTS ---
# Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- 3. DATA LOADER & VECTOR STORE ---
def get_vectorstore():

    if os.path.exists(DB_PERSIST_DIR) and os.listdir(DB_PERSIST_DIR):
        console.print("[green]Loading existing Vector Database from disk...[/green]")
        return Chroma(persist_directory=DB_PERSIST_DIR, embedding_function=embeddings)
    
    console.print("[yellow]Index not found. Building Vector Database from JSONs...[/yellow]")
    documents = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not json_files:
        console.print(f"[red]No JSON files found in {DATA_FOLDER}[/red]")
        exit()

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
                
                for entry in data:
                    meta = entry.get('metadata', {})
                    # Fallback 
                    content = entry.get('content') or ""
                    if not content: continue
                    
                    clean_meta = {
                        "source": meta.get('source_url') or "Unknown",
                        "program_name": meta.get('program_name') or "General Info",
                        "type": meta.get('document_type') or "Doc",
                        "page": str(meta.get('page_number') or "N/A")
                    }
                    documents.append(Document(page_content=content, metadata=clean_meta))
        except Exception as e:
            console.print(f"Error loading {file_path}: {e}")

    # create db
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_PERSIST_DIR
    )
    console.print(f"[bold green]Indexed {len(documents)} chunks into Vector Store.[/bold green]")
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 

# --- 4. MEMORY & CHAINS ---
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

rephrase_system = """Given a chat history and the latest user question which might reference context 
in the chat history, formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed or otherwise return it as is."""

rephrase_chain = (
    ChatPromptTemplate.from_messages([
        ("system", rephrase_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]) 
    | llm 
    | StrOutputParser()
)

rag_template = """You are TULens, an expert assistant for TU Wien.
Answer based ONLY on the context below.

RULES:
1. If context is empty, say "I don't have that information."
2. For lists/requirements, use Bullet Points.
3. Cite sources at the end: [Source: Document Name (Page X)].

CONTEXT:
{context}

QUESTION: 
{question}
"""
rag_chain = (
    ChatPromptTemplate.from_messages([("human", rag_template)])
    | llm
    | StrOutputParser()
)

# --- 5. HELPER FUNCTIONS ---
def format_docs_with_metadata(docs):
    formatted = []
    for doc in docs:
        meta = doc.metadata
        header = f"SOURCE: {meta.get('program_name')} | TYPE: {meta.get('type')}"
        formatted.append(f"--- {header} ---\n{doc.page_content}\n")
    return "\n".join(formatted)

# --- 6. MAIN LOOP ---
console.print("\n[bold green]TULens Demo Ready![/bold green] (Type 'exit' to stop)\n")
session_id = "demo_session"

while True:
    try:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")
        if user_input.lower() in ["exit", "quit"]: break
        if not user_input.strip(): continue

        with console.status("[bold yellow]Thinking & Retrieving...[/bold yellow]", spinner="dots"):
            # 1. Contextualize Question
            history = get_session_history(session_id)
            standalone_question = rephrase_chain.invoke({
                "chat_history": history.messages,
                "question": user_input
            })
            
            # 2. Retrieve
            relevant_docs = vectorstore.similarity_search(standalone_question, k=5)
            
            # 3. Generate
            if not relevant_docs:
                response = "I don't have that information in my database."
            else:
                formatted_context = format_docs_with_metadata(relevant_docs)
                response = rag_chain.invoke({
                    "question": standalone_question, 
                    "context": formatted_context
                })
                
                # Update Memory
                history.add_user_message(user_input)
                history.add_ai_message(response)

        console.print(Panel(Markdown(response), title="TULens Bot", border_style="green"))

        # 4. Sources Display
        if relevant_docs:
            console.print("[italic dim]Sources used:[/italic dim]")
            seen_sources = set()
            for doc in relevant_docs:
                meta = doc.metadata
                prog_name = meta.get('program_name', 'Doc')
                if len(prog_name) > 40: prog_name = prog_name[:37] + "..."
                page = meta.get('page', 'N/A')
                location = f"Page {page}" if page and page != "N/A" else "Web"
                
                source_entry = f"{prog_name} ({location})"
                if source_entry not in seen_sources:
                    console.print(f"   • {source_entry}", style="dim")
                    seen_sources.add(source_entry)
            console.print("")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")