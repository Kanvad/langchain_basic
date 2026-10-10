from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are a helpful and very concise assistant.'

    match user_role:
        case 'expert':
            return f"{base_prompt} Provide detailed and technical explanations."
        case 'beginner':
            return f"{base_prompt} Keep explanations simple and easy to understand."
        case 'child':
            return f"{base_prompt} Explain everything as if you were literally talking to a five-year-old child."
        case _:
            return base_prompt

agent = create_agent(
    model = 'openrouter:nvidia/nemotron-3.5-lightning:free',
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain how a car engine works."}],},
    context=Context(user_role="child"))

print(response)