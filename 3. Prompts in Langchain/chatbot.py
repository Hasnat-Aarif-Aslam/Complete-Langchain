from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use conversational task (important!)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="conversational",   
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=128,
    temperature=0.1,
)

# Wrap inside ChatHuggingFace for chatbot-like behavior
chat_model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot...")
        break
    response = chat_model.invoke(user_input)
    print("Bot:", response.content)
