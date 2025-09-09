from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=55)

document = [
    'where are you from?',
    'what is your name?',
    'what is your quest?'
]

doc_result = embeddings.embed_documents(document)
print(str(doc_result))