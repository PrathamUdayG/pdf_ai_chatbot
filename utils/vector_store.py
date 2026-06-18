import chromadb
from sentence_transformers import SentenceTransformer
# ---------------------------------
# Load Embedding Model Once
# ---------------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
# ---------------------------------
# Persistent Chroma Client
# ---------------------------------
client = chromadb.PersistentClient(
    path="./chroma_db"
)
# ---------------------------------
# Create Collection
# ---------------------------------
collection = client.get_or_create_collection(
    name="pdf_collection"
)
# ---------------------------------
# Clear Existing Collection
# ---------------------------------
def clear_collection():
    global collection
    try:
        client.delete_collection(
            "pdf_collection"
        )
    except:
        pass
    collection = client.get_or_create_collection(
        name="pdf_collection"
    )
# ---------------------------------
# Store Chunks
# ---------------------------------
def store_chunks(chunks):
    texts = []
    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        texts.append(
            chunk["text"]
        )
        ids.append(
            f"{chunk['source']}_"
            f"{chunk['page']}_"
            f"{i}"
        )
        metadatas.append(
            {
                "page": str(
                    chunk["page"]
                ),
                "source": chunk["source"]
            }
        )
    # Batch embeddings
    embeddings = model.encode(
        texts
    ).tolist()
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
# ---------------------------------
# Retrieve Chunks
# ---------------------------------
def retrieve_chunks(
        query,
        k=3
):
    query_embedding = model.encode(
        query
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results