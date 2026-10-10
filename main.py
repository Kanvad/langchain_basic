from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

basic_model = init_chat_model(
    model = 'openrouter:nvidia/nemotron-3.5-lightning:free'
)
advanced_model = init_chat_model(
    model = 'openrouter:liquid/lfm-2.5-2.6b:free'
)

@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count < 3:
        model = basic_model
    else:
        model = advanced_model
    
    request.model = model
    return handler(request)

agent = create_agent(
    model = basic_model,
    middleware=[dynamic_model_selector]
)

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is 1 + 1?"),],
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])