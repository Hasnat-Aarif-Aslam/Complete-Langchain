from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=55)
query_result = embeddings.embed_query("Hello world")
print(str(query_result))