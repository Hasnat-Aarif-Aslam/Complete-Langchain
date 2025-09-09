from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=64,
    temperature=0.0,
    do_sample=False,
    return_full_text=False,   # only return the model’s answer, not your prompt
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Pakistan?")
print(result.content)