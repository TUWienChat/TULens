import os

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, CHROMADB_COLLECTION

# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = "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"
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", 8000)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Connect to ChromaDB
client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

vectorstore = Chroma(
    client=client,
    collection_name=CHROMADB_COLLECTION,
    embedding_function=embeddings
)


# Create retriever with configurable k value
def create_retriever(k=3):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def print_document_details(doc, index):
    """Print detailed information about a single document."""
    print(f"\n{'=' * 80}")
    print(f"📄 DOCUMENT {index}")
    print('=' * 80)

    # Content
    print("\n📝 CONTENT:")
    print("-" * 80)
    print(doc.page_content)
    print("-" * 80)

    # Metadata
    print("\n🏷️  METADATA:")
    if doc.metadata:
        for key, value in doc.metadata.items():
            print(f"  • {key}: {value}")
    else:
        print("  No metadata available")

    # Similarity score if available
    if hasattr(doc, 'score'):
        print(f"\n🎯 SIMILARITY SCORE: {doc.score}")


def print_summary(docs):
    """Print a summary of all retrieved documents."""
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: Retrieved {len(docs)} documents")
    print("=" * 80)

    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        source = doc.metadata.get('source', 'Unknown source')
        print(f"\n{i}. {source}")
        print(f"   Preview: {preview}...")


def inspect_documents(query, k=3, show_full=True):
    """Retrieve and inspect documents for a given query."""
    print(f"\n🔍 SEARCHING FOR: '{query}'")
    print(f"📌 Retrieving top {k} documents...\n")

    retriever = create_retriever(k=k)
    docs = retriever.invoke(query)

    if not docs:
        print("⚠️  No documents found!")
        return

    # Show summary first
    print_summary(docs)

    # Show full details if requested
    if show_full:
        print("\n" + "=" * 80)
        print("📖 FULL DOCUMENT DETAILS")
        print("=" * 80)
        for i, doc in enumerate(docs, 1):
            print_document_details(doc, i)

    print("\n" + "=" * 80 + "\n")


def search_with_scores(query, k=3):
    """Search and return documents with similarity scores."""
    print(f"\n🔍 SEARCHING WITH SCORES FOR: '{query}'")
    print(f"📌 Retrieving top {k} documents...\n")

    # Use similarity_search_with_score for scores
    results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        print("⚠️  No documents found!")
        return

    print("=" * 80)
    print(f"📊 FOUND {len(results)} DOCUMENTS WITH SCORES")
    print("=" * 80)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Document {i} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:150]}...")

    print("\n" + "=" * 80 + "\n")


# --- MAIN INTERACTIVE LOOP ---
def main():
    print("\n" + "=" * 80)
    print("🔍 VECTOR DATABASE DOCUMENT INSPECTOR")
    print("=" * 80)
    print("\nCommands:")
    print("  • Type your search query to inspect documents")
    print("  • 'summary' - Show only document summaries (no full content)")
    print("  • 'scores' - Search with similarity scores")
    print("  • 'k=N' - Set number of documents to retrieve (e.g., 'k=5')")
    print("  • 'exit' or 'quit' - Exit the inspector")
    print("\n" + "=" * 80 + "\n")

    k_value = 3
    show_full = True

    while True:
        try:
            user_input = input("🔍 Search query: ").strip()

            # Exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Toggle summary mode
            if user_input.lower() == "summary":
                show_full = not show_full
                mode = "FULL" if show_full else "SUMMARY ONLY"
                print(f"📋 Display mode: {mode}\n")
                continue

            # Change k value
            if user_input.lower().startswith("k="):
                try:
                    k_value = int(user_input.split("=")[1])
                    print(f"✅ Set k={k_value}\n")
                except ValueError:
                    print("❌ Invalid k value. Use format: k=5\n")
                continue

            # Search with scores
            if user_input.lower() == "scores":
                query = input("Enter query for score search: ").strip()
                if query:
                    search_with_scores(query, k=k_value)
                continue

            # Regular document inspection
            inspect_documents(user_input, k=k_value, show_full=show_full)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()