import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description="Get the current weather for a given city", return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

openrouter = ChatOpenRouter(model="nvidia/nemotron-3.5-lightning:free")

agent = create_agent(
    model=openrouter,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can provide weather information for any city, who always cracks jokes about the weather and is very funny, but also provides accurate information.",
)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather like in Ho Chi Minh City?"}
        ]
    }
)

print(response)
print(response["messages"][-1].content)