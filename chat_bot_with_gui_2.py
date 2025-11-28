from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
import gradio as gr
from dotenv import load_dotenv
from os import getenv, path

load_dotenv()
api_key = getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

if path.exists("mall_db"):
    db = Chroma(persist_directory="mall_db", embedding_function=embeddings)
else:
    loader = CSVLoader("products.csv")
    documents = loader.load()
    db = Chroma.from_documents(documents, embeddings, persist_directory="mall_db")

retriever = db.as_retriever(search_kwargs={"k": 3})

chat = chat = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

response_template = """
أنت مساعد ذكي في مول إلكتروني.
استخدم المحادثة السابقة لفهم سياق المستخدم.
إذا لم تجد الإجابة في معلومات المنتجات قم بالإجابة من نفسك

 المحادثة السابقة:
{response_chat_history}

 معلومات المنتجات:
{context}

 سؤال المستخدم:
{question}

"""

response_prompt = PromptTemplate(
    template=response_template,
    input_variables=["response_chat_history", "context", "question"],
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

response_chain = response_prompt | chat
json_chain = json_prompt | chat | parser
combined_chain = RunnableParallel({
    "response": response_chain,
    "json": json_chain
})

def mall_bot(message, history):
    question = message.strip()
    if not question: return ""

    response_chat_history = response_memory.load_memory_variables({}).get("response_chat_history", "")
    json_chat_history = json_memory.load_memory_variables({}).get("json_chat_history", "")

    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    answer = combined_chain.invoke({
        "response_chat_history": response_chat_history,
        "json_chat_history": json_chat_history,
        "context": context,
        "question": question
    })

    # الرمز ده (u\200f) هو حرف غير مرئي بيجبر السطر يبقى يمين
    rtl_mark = "\u200f" 
    
    json_items_str = "\n".join(
        # ضفنا الرمز قبل كل سطر
        [f"{rtl_mark}- {item.product} : {item.price} LE ({item.store})" for item in answer["json"].items]
    )
    
    json_memory.save_context(
        {"input": question},
        {"output": json_items_str}
    )

    response_memory.save_context(
        {"input": question},
        {"output": answer["response"].content}
    )

    return answer["response"].content

rtl_css = """
.message {
    text-align: right !important;
    direction: rtl !important;
}
/* تظبيط القوائم (النقط) عشان تيجي على اليمين */
ul {
    direction: rtl !important;
    text-align: right !important;
}
li {
    direction: rtl !important;
    text-align: right !important;
}
"""

demo = gr.ChatInterface(
    fn=mall_bot,
    title="Mall Assistant Bot 🛒",
    description="مساعد ذكي للمول، اسأل عن المنتجات والأسعار.",
    examples=["سعر الايفون كام؟", "عندكم لابتوب ديل؟", "ايه أرخص شاشة؟"],
    theme="soft",
    type="messages",
    css=rtl_css 
)

if __name__ == "__main__":
    demo.launch()