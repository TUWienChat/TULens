import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
# Get your free key from console.groq.com
os.environ["GROQ_API_KEY"] = "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"

# 1. SETUP MODEL & EMBEDDINGS
# We use HuggingFace (Local/Free) for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We use Groq (Llama 3) for the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


json_files = [
    "tuwien_informatics_scrape_msc_demo.json",
    "tuwien_informatics_pdf_demo.json"
]
# 2. LOAD DATA
# Using your spider's output format
all_docs = []
for json_file in json_files:
    loader = JSONLoader(
        file_path=json_file,
        jq_schema='.[]',
        content_key='content',
        metadata_func=lambda record, meta: {**meta, "source": record.get("url")}
    )
    all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# # 3. VECTOR STORE
# # This works without importing 'langchain.chains'
# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     # persist_directory="./chroma_db"
# )
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. DEFINE THE PROMPT
template = """You are a helpful assistant for TU Wien.
Answer the question based only on the following context:

{context}

Question: {question}
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