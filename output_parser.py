from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
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

schemas = [
    ResponseSchema(name="product", description="اسم المنتج"),
    ResponseSchema(name="price", description="سعر المنتج"),
    ResponseSchema(name="store", description="اسم المتجر")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

template = """
أنت مساعد ذكي في مول إلكتروني. جاوب بصيغة JSON فقط.
استخدم المعلومات التالية:

{context}

سؤال المستخدم: {question}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | chat | parser

user_question = "فين أقدر أشتري حذاء Nike؟"
context = get_context(user_question)

result = chain.invoke({"context": context, "question": user_question})
print(result)