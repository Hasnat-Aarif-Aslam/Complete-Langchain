from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} assistant."),
    ("human", "what is {topic}? explain in a 2 sentences."),
])

prompt = chat_template.invoke({"domain": "AI", "topic": "langchain"})
print(prompt)