import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model="openrouter:nvidia/nemotron-3.5-lightning:free",
    temperature=0.1,
)

for chunk in model.stream("what is python?"):
    print(chunk.text, end="", flush=True)