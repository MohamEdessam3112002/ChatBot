from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = CSVLoader("products.csv")
documents = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma.from_documents(documents, embeddings, persist_directory="mall_db")

query = "فين أقدر أشتري حذاء Nike؟"
results = db.similarity_search(query, k=1)

for result in results:
    print(result.page_content)