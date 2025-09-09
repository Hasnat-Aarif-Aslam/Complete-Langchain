from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=55)

document = [
    'i saw a man walking his cat in the park',
    'the weather is sunny today',
    'my favorite food is pizza'
    'have you ever been to New York City?',
    'todays weather is cold and rainy',  # query should match this
    
]

query = 'tell me about todays weather'


doc_embed = embeddings.embed_documents(document)
query_embed = embeddings.embed_query(query)

similarity = cosine_similarity([query_embed], doc_embed)[0]
print(similarity)

print(sorted(list(enumerate(similarity)), key=lambda x: x[1]))




