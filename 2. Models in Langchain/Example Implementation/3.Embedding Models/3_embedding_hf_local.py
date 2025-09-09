from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = 'How are you?'

query_result = embedding.embed_query(text)
print(str(query_result))