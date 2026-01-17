import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
# Get your free key from console.groq.com
os.environ["GROQ_API_KEY"] = "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"


# We use Groq (Llama 3) for the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.HttpClient(host="localhost", port=8000)

vectorstore = Chroma(
    client=client,
    collection_name="tu_wien_informatics",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. DEFINE THE PROMPT
template = """You are a helpful assistant for TU Wien master students.
Be as concise as possible and follow the following rules:
Rule 1: Answer always in english.
Rule 2: Answer based on context
Rule 3: cite the source URL or PDF name for every claim you make

Here is the provided context:
{context}

Please answer the following question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. BUILD THE PURE LCEL CHAIN (No 'langchain.chains' module needed)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. RUN
print("\n✅ TUWienChat Demo Ready! (Type 'exit' or 'quit' to stop)\n")

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

        # Run the chain
        response = rag_chain.invoke(user_input)

        # Print result
        print(f"Bot: {response}\n")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")