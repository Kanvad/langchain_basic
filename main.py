import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="liquid/lfm-2.5-embedding-350m:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

texts = [
    "Apple makes very good computers.",
    "I believe Apple is innovative.",
    "I love apples.",
    "I am a fan of MacBooks.",
    "I enjoy oranges.",
    "I Like Lenovo Thinkpads.",
    "I think pears taste very good.",
]

vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# print(vectorstore.similarity_search("Apples is my favorite food.", k=7))
print(vectorstore.similarity_search("Linux is a good operating system.", k=7))