from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0)

tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is 12323 times 3234234235? ")

##########################################################################################################

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What year was Vladimir Putin born? What is that year multiplied by 2?")

##########################################################################################################

from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent

agent = create_python_agent(tool=PythonREPLTool(), llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
py_list = [1, 23, 5324, 3, 23, 526, 7, 43]
agent.run(f"Sort this Python list: {py_list}")

##########################################################################################################

from langchain.agents import tool


@tool
def smartest_person(text: str) -> str:
    """
    Returns the name of the smartest person in the universe.
    Expects an input of an empty string and returns the name.
    """
    return "Albert Einstein"


llm = ChatOpenAI(temperature=0)
tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
tools += [smartest_person]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who's the smartest person in the universe?")

##########################################################################################################

from langchain.agents import tool
from datetime import datetime


@tool
def get_current_time(text: str) -> str:
    """
    Returns the current time. Use this for any question regarding the current time or date.
    Input is an empty string and the current time is returned in a string format. Only use this
    function for the current time, other time related functions should use another tool.
    """
    return str(datetime.now())


llm = ChatOpenAI(temperature=0)
tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
tools += [get_current_time]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is the current time?")

##########################################################################################################

from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0)
tools = load_tools(['llm-math'], llm=llm)
memory = ConversationBufferMemory(memory_key='chat_history')
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
print(agent.run(input="How to I earn money with LLMs and Langchain?"))
print(agent.run(input="What's the fastest and easiest method out of those?"))
