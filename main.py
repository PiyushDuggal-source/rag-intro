import ollama
import os
import chromadb

import chromadb.utils.embedding_functions as embedding_functions


# def create_embeddings(texts, collection_name="my_collection"):
#     # Initialize ChromaDB client
#     client = chromadb.PersistentClient(path="./chroma_db")
#
#     # Create or get a collection
#     collection = client.get_or_create_collection(name=collection_name)
#
#     # Generate embeddings using Ollama
#     embeddings = [
#         ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
#         for text in texts
#     ]
#
#     # Add documents and embeddings to ChromaDB
#     for i, text in enumerate(texts):
#         collection.add(ids=[str(i)], documents=[text], embeddings=[embeddings[i]])
#
#     print(f"Added {len(texts)} documents to collection '{collection_name}'.")
#     return collection
#
#
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="my_collection", embedding_function=ollama_ef
)


def get_embeddings(text):
    resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return resp.embedding


def get_ollama_res(prompt, system):

    for part in ollama.generate(
        model="llama3.2", prompt=prompt, system=system, stream=True
    ):
        print(part["response"], end="", flush=True)
    # Function to load documents from a directory def load_documents_from_directory(directory_path):


def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# load documents

dir_path = "./docs"

documents = load_documents_from_directory(dir_path)

chunked_docs = []

for doc in documents:
    chunk = split_text(doc["text"])
    print("==== Spliting documents ====")
    for i, chunk in enumerate(chunk):
        chunked_docs.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# create embeddings
for doc in chunked_docs:
    print("==== Creating embeddings ====")
    doc["embeddings"] = get_embeddings(doc["text"])


# for doc in chunked_docs:
#     print("==== Inserting chunk into db ====")
#
#     collection.upsert(
#         ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embeddings"]]
#     )


# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

    return relevant_chunks


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    get_ollama_res(question, prompt)


question = "What did aman do in the mars?"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
