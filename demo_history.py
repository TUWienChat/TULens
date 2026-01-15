import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever

from config import LLM_MODEL, EMBEDDING_MODEL, CHROMADB_COLLECTION

# --- CONFIGURATION ---
# Get your free key from console.groq.com
os.environ["GROQ_API_KEY"] = "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"


# We use Groq (Llama 3) for the LLM
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

client = chromadb.HttpClient(host="localhost", port=8000)

vectorstore = Chroma(
    client=client,
    collection_name=CHROMADB_COLLECTION,
    embedding_function=embeddings
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# bm25_retriever = BM25Retriever.from_documents(all_splits)
# bm25_retriever.k = 5
#
# # Combine both (60% vector, 40% keyword)
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[vector_retriever, bm25_retriever],
#     weights=[0.6, 0.4]
# )

# 4. DEFINE THE PROMPT WITH CHAT HISTORY
template = """You are a helpful assistant for TU Wien master students.
Be as concise as possible and follow the following rules:
Rule 1: Answer always in english.
Rule 2: Answer based on context
Rule 3: Cite the source URL or PDF name for every claim you make

Here is the provided context:
{context}

Please answer the following question:
{question}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 5. BUILD THE LCEL CHAIN WITH CHAT HISTORY
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_chain_input(input_dict):
    """Prepare the input for the chain with all required variables."""
    return {
        "context": input_dict["context"],
        "question": input_dict["question"],
        "chat_history": input_dict["chat_history"]
    }

# Build the chain
rag_chain = (
    {
        "context": lambda x: format_docs(vector_retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. RUN WITH CHAT HISTORY
print("\n✅ TUWienChat Demo Ready with Chat History! (Type 'exit' or 'quit' to stop)\n")

# Initialize chat history
chat_history = []

while True:
    try:
        # Get user input
        user_input = input("You: ")

        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Skip empty inputs
        if not user_input.strip():
            continue

        print("Thinking...")

        # Run the chain with chat history
        response = rag_chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })

        # Print result
        print(f"Bot: {response}\n")

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        # Optional: Limit chat history to last N messages to avoid context overflow
        # For example, keep only last 10 messages (5 exchanges)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")