from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os


chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} assistant. You must review the chat history before answering the user query. Your answer/response must be according to the previous conversation.If you don't know the answer, just say that you don't know, don't try to make up an answer, and always be concise in your response."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

# 2) Parse your file into role-aware messages
history_path = r"C:\Users\HasnatAA\Downloads\New folder (35)\4. Messages in Langchain\chat_history.txt"

def load_history(path: str):
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Case A: lines exactly like HumanMessage(content="...") / AIMessage(content="...")
            if line.startswith('HumanMessage(content="') and line.endswith('")'):
                content = line[len('HumanMessage(content="'):-2]
                msgs.append(HumanMessage(content=content))
                continue
            if line.startswith('AIMessage(content="') and line.endswith('")'):
                content = line[len('AIMessage(content="'):-2]
                msgs.append(AIMessage(content=content))
                continue

            # Case B (fallback): role: text
            if ":" in line:
                role, content = line.split(":", 1)
                role = role.strip().lower()
                content = content.strip().strip('"')
                if role in ("user", "human"):
                    msgs.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    msgs.append(AIMessage(content=content))
                else:
                    raise ValueError(f"Unknown role: {role}")
            else:
                raise ValueError(f"Unrecognized history line: {line}")
    return msgs

chat_history = load_history(history_path)



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
    
    prompt = chat_template.invoke({
    "domain": "e-commerce",
    "chat_history": chat_history,
    "query": user_input
    })

    response = chat_model.invoke(prompt)
    chat_history.append(HumanMessage(content = user_input))
    chat_history.append(AIMessage(content = response.content))
    print("Bot:", response.content)
    print(chat_history)

print("Chat session ended. Here is the full chat history:", chat_history)