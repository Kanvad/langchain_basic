import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="openrouter:nvidia/nemotron-3.5-lightning:free",
    temperature=0.1,
)

response = model.invoke("Hello, what is javascript?")

print(response)
print(response.content)