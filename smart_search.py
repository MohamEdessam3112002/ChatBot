from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = CSVLoader("products.csv")
documents = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma.from_documents(documents, embeddings, persist_directory="mall_db")

chat = chat = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

retriever = db.as_retriever(search_kwargs={"k":1})

def get_context(question):
    docs = retriever._get_relevant_documents(question, run_manager=None)
    context = "\n".join([doc.page_content for doc in docs])
    return context

template = """
أنت مساعد ذكي في مول إلكتروني. جاوب المستخدم بدقة وبأسلوب بسيط.
استخدم المعلومات التالية فقط:

{context}

سؤال المستخدم: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
chain = prompt | chat

user_question = "فين أقدر أشتري حذاء Nike؟"
context = get_context(user_question)

result = chain.invoke({"context": context, "question": user_question})
print(result.content)