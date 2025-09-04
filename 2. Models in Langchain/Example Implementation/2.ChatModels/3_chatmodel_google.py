from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

result = chat_model.invoke("Write a poem about the Langchain library")

print(result.content)