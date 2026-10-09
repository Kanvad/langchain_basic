import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model="openrouter:nvidia/nemotron-3.5-lightning:free",
    temperature=0.1,
)

conversation = [
    SystemMessage(content="You are a helpful assistant that can provide information about programming languages."),
    HumanMessage(content="Hello, what is javascript?"),
    AIMessage(content="JavaScript is a versatile programming language primarily used for web development"),
    HumanMessage(content="Can you give me an example of a simple JavaScript function?")
]

response = model.invoke(conversation)

print(response)
print(response.content)