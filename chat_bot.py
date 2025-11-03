from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("GOOGLE_API_KEY")

loader = CSVLoader("products.csv")
documents = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma.from_documents(documents, embeddings, persist_directory="mall_db")
retriever = db.as_retriever(search_kwargs={"k":3})

chat = chat = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

response_template = """
أنت مساعد ذكي في مول إلكتروني.
استخدم المحادثة السابقة لفهم سياق المستخدم.
إذا لم تجد 

 المحادثة السابقة:
{response_chat_history}

 معلومات المنتجات:
{context}

 سؤال المستخدم:
{question}

"""

response_prompt = PromptTemplate(
    template=response_template,
    input_variables=["json_chat_history", "context", "question"],
)

response_memory = ConversationBufferMemory(memory_key="response_chat_history", return_messages=False)

class ProductInfo(BaseModel):
    product: str = Field(description = "اسم المنتج")
    price: float = Field(description="سعر المنتج")
    store: str = Field(description="اسم المتجر")

class ProductList(BaseModel):
    items: list[ProductInfo]

parser = PydanticOutputParser(pydantic_object=ProductList)
format_instructions = parser.get_format_instructions()

json_template = """
أنت مساعد ذكي في مول إلكتروني.
استخدم المحادثة السابقة لفهم سياق المستخدم.
جاوب بصيغة JSON فقط.

 المحادثة السابقة:
{json_chat_history}

 معلومات المنتجات:
{context}

 سؤال المستخدم:
{question}

{format_instructions}
"""

json_prompt = PromptTemplate(
    template=json_template,
    input_variables=["json_chat_history", "context", "question"],
    partial_variables={"format_instructions": format_instructions},
)

json_memory = ConversationBufferMemory(memory_key="json_chat_history", return_messages=False)

while True:
    question = input("أنت: ").strip()
    if question.lower() in ["خروج", "exit", "quit"]:
        break
    
    response_chat_history = response_memory.load_memory_variables({}).get("response_chat_history", "")
    json_chat_history = json_memory.load_memory_variables({}).get("json_chat_history", "")

    docs = retriever._get_relevant_documents(question, run_manager=None)
    context = "\n".join([d.page_content for d in docs])

    response_chain = response_prompt | chat
    json_chain = json_prompt | chat | parser
    combined_chain = RunnableParallel({
        "response": response_chain,
        "json": json_chain
    })

    answer = combined_chain.invoke({
        "response_chat_history": response_chat_history,
        "json_chat_history": json_chat_history,
        "context": context,
        "question": question
    })

    print("بوت:", answer["response"].content)
    print("\n\n")
    print("json:", answer["json"].items)

    json_memory.save_context(
    {"input": question},
    {
        "output": "\n".join(
            [f"{item.product} - {item.price} - {item.store}" for item in answer["json"].items]
        )
    }
    )

    response_memory.save_context(
    {"input": question},
    {"output": answer["response"].content}
    )