from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

template = "أنت مساعد ذكي في مول. جاوب على السؤال التالي بالعربية: {question}"
prompt = PromptTemplate.from_template(template)

chat = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

chain = prompt | chat

response = chain.invoke({"question": "هل في خصومات على الموبايلات؟"})
print(response.content)