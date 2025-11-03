from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

messages = [
    SystemMessage(content="أنت مساعد ذكي في مول إلكتروني، تساعد الزوار في إيجاد المنتجات والعروض."),
    HumanMessage(content="فين أقدر أشتري حذاء Nike؟")
]

response = chat.invoke(messages)
print(response.content)