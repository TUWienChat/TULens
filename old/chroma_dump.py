import chromadb
import json

# Connect to your Dockerized or Local Chroma
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("tu_wien_informatics")

# Fetch EVERYTHING (documents, metadatas, and embeddings)
data = collection.get()

# Dump to a JSON file
with open("chroma_dump.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"Dumped {len(data['ids'])} records to chroma_dump.json")