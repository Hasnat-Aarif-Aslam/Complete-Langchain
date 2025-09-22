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

# Maintaining chat history so our model has context
chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot...")
        break
    response = chat_model.invoke(chat_history)
    chat_history.append(response.content) # we have appended both user and bot messages, but we havent specified who said what, and this is an issue.
    print("Bot:", response.content)
