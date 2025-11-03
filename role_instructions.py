from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد ذكي متخصص في تسويق منتجات المول."),
    ("user", "اشرحلي فائدة منتجات العناية بالبشرة بأسلوب بسيط.")
])

response = chat.invoke(prompt.format())
print(response.content)

print("_______________________________________________________")

#Response Format
prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد تسويق."),
    ("user", "اعمل مقارنة بين منتجين للعناية بالبشرة بصيغة JSON، تشمل السعر، المميزات، والعيوب.")
])

response = chat.invoke(prompt.format())
print(response.content)

print("_______________________________________________________")

#Tone Style
prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد يتحدث بأسلوب مرح وودود."),
    ("user", "اكتب إعلان قصير لمنتج عطور جديدة في المول.")
])

print("_______________________________________________________")

#Variable user prompt
response = chat.invoke(prompt.format())
print(response.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد يتحدث بأسلوب مرح وودود."),
    ("user", "{question}")
])
question = "ايه أفضل عطر مناسب للرجال في الصيف؟"

response = chat.invoke(prompt.format(question=question))
print(response.content)