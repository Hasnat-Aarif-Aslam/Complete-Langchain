from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot...")
        break
    response = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content = response.content))
    print("Bot:", response.content)

print("Chat session ended. Here is the full chat history:", chat_history)