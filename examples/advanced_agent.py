from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    termperature_celsius: float
    termperature_fahrenheit: float
    humidity: float

@tool('get_weather', description="Get the current weather for a given city", return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

@tool('locate_user', description="Look up a user base on the context", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "user_1":
            return "Ho Chi Minh City"
        case "user_2":
            return "New York City"
        case _:
            return "Unknown"

model = init_chat_model(
    model="openrouter:nvidia/nemotron-3.5-lightning:free",
    temperature=0.3,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a funny and accurate weather assistant for any city.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = { 'configurable': {'thread_id': 1} }

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather like?"}
        ]},
        config = config,
        context = Context(user_id="user_1"),
)

# print(response)
print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].termperature_celsius)