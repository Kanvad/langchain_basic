import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="liquid/lfm-2.5-embedding-350m:free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

texts = [
    "I love apples.",
    "I enjoy oranges. ",
    "I think pears taste very good.",
    "I hate bananas. ",
    "I dislike raspberries.",
    "I despise mangos.",
    "I love Linux.",
    "I hate Windows.",
]

vectorstore = FAISS.from_texts(texts, embedding=embeddings)

print(vectorstore.similarity_search("What fruits does the person like?", k=3))
print(vectorstore.similarity_search("What fruits does the person hate?", k=3))

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search small product / fruit knowledge base for information.",
)

agent = create_agent(
    model="openrouter:openrouter/free",
    tools=[retriever_tool],
    system_prompt=("You are a helpful assistant that answers questions about Macs, apples or Laptops, first call the kb_search tool to retrieve context, then answer the question based on the retrieved information. Maybe you have to use it multiple times before you can answer the question."),
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What fruits does the person like and what fruits does the person dislike?"},
    ]
})

print(result)
print(result["messages"][-1].content)