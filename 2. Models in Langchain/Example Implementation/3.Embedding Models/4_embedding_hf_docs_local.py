from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    'How are you?',
    'What is your name?',
    'Where do you live?'
]

query_result = embedding.embed_documents(documents)
print(str(query_result))