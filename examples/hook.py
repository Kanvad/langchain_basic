import time
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

class HooksDemo(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.start_time = 0.0
    
    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("Before agent triggered")
    
    def before_model(self, state: AgentState, runtime):
        print("Before model")
    
    def after_model(self, state: AgentState, runtime):
        print("After model")
    
    def after_agent(self, state: AgentState, runtime):
        print("After agent: ", time.time() - self.start_time)

agent = create_agent('openrouter:nvidia/nemotron-3.5-lightning:free', middleware=[HooksDemo()])

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is Python?"),],
})

print(response["messages"][-1].content)