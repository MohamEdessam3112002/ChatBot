from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
text = "حذاء Nike Air Max خفيف ومريح"
vector = embeddings.embed_query(text)
print(len(vector))
print(vector[:10])