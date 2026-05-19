from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


loader = PyPDFLoader("/Users/rahulmandal/self-healing-rag/data/raw/infosys_annual_report_2024.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")


for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="tenk_corpus",
    persist_directory="data/chroma_db"
)

print("\nDone! Chunks embedded and stored in ChromaDB.")


query = "Infosys revenue 2024"
results = vectorstore.similarity_search_with_score(query, k=3)

print("\n--- Retrieval Test ---")
for doc, score in results:
    print(f"\nScore: {score:.4f}")
    print(doc.page_content[:200])