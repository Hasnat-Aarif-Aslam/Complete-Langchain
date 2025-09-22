# There are 3 types of messages:
# 1. HumanMessage: Represents a message from a human user.
# 2. AIMessage: Represents a message from an AI model.
# 3. SystemMessage: Represents a system-level message, often used for instructions or context (for example: You are a helpful assistant).

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="what is langchain? explain in a single sentence."),
]

result = chat_model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)