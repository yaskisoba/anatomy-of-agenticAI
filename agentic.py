import functools
import os
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import io
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from IPython.display import display
from PIL import Image
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

model = ChatOpenAI(temperature=0, streaming=True)


def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  # We return a list, because this will get added to the existing list
  return {"messages": [response]}


def sense(state):
  pass


def reason(state):
  pass


def plan(state):
  pass


def coordinate(state):
  pass


def act(state):
  pass


def use_tools(state):
  pass


def memory(state):
  pass


def goals(state):
  pass


def create_agent():
  workflow = StateGraph(AgentState)
  workflow.add_node("agent", call_model)
  workflow.add_node("goals", goals)
  workflow.add_node("sense", sense)
  workflow.add_node("plan", plan)
  workflow.add_node("coordinate", coordinate)
  workflow.add_node("act", act)
  workflow.add_node("tools", use_tools)
  workflow.add_node("reason", reason)
  workflow.add_node("memory", memory)

  workflow.set_entry_point("agent")  # primeiro nó a ser chamado

  workflow.add_edge("agent", "goals")
  workflow.add_edge("agent", "plan")
  workflow.add_edge("plan", "coordinate")
  workflow.add_edge("agent", "sense")
  workflow.add_edge("agent", "memory")
  workflow.add_edge("coordinate", "act")
  workflow.add_edge("memory", "agent")
  workflow.add_edge("agent", "reason")
  workflow.add_edge("act", "tools")
  workflow.set_finish_point("act")

  app = workflow.compile()

  return app


class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]


def agentic():
  # agent_1 = create_agent()
  # agent_1_node = functools.partial()

  #grafo do agente 1
  agent_1 = StateGraph(AgentState)
  agent_1.add_node("agent", call_model)
  agent_1.add_node("goals", goals)
  agent_1.add_node("sense", sense)
  agent_1.add_node("plan", plan)
  agent_1.add_node("coordinate", coordinate)
  agent_1.add_node("act", act)
  agent_1.add_node("tools", use_tools)
  agent_1.add_node("reason", reason)
  agent_1.add_node("memory", memory)

  agent_1.set_entry_point("agent")  # primeiro nó a ser chamado

  agent_1.add_edge("agent", "goals")
  agent_1.add_edge("agent", "plan")
  agent_1.add_edge("plan", "coordinate")
  agent_1.add_edge("agent", "sense")
  agent_1.add_edge("agent", "memory")
  agent_1.add_edge("coordinate", "act")
  agent_1.add_edge("memory", "agent")
  agent_1.add_edge("agent", "reason")
  agent_1.add_edge("act", "tools")
  agent_1.set_finish_point("act")

  # grafo do agente 2
  agent_2 = StateGraph(AgentState)
  agent_2.add_node("agent", call_model)
  agent_2.add_node("goals", goals)
  agent_2.add_node("sense", sense)
  agent_2.add_node("plan", plan)
  agent_2.add_node("coordinate", coordinate)
  agent_2.add_node("act", act)
  agent_2.add_node("tools", use_tools)
  agent_2.add_node("reason", reason)
  agent_2.add_node("memory", memory)

  agent_2.set_entry_point("agent")  # primeiro nó a ser chamado

  agent_2.add_edge("agent", "goals")
  agent_2.add_edge("agent", "plan")
  agent_2.add_edge("plan", "coordinate")
  agent_2.add_edge("agent", "sense")
  agent_2.add_edge("agent", "memory")
  agent_2.add_edge("coordinate", "act")
  agent_2.add_edge("memory", "agent")
  agent_2.add_edge("agent", "reason")
  agent_2.add_edge("act", "tools")
  agent_2.set_finish_point("act")

  #grafo da interação entre agentes
  workflow = StateGraph(AgentState)
  workflow.add_node("agent_1", agent_1.compile())
  workflow.set_entry_point("agent_1")  # primeiro nó a ser chamado
  workflow.add_node("agent_2", agent_2.compile())

  workflow.add_edge('agent_1', 'agent_2')
  workflow.add_edge('agent_2', 'agent_1')

  app = workflow.compile()

  try:
    image_data = app.get_graph(xray=True).draw_png()
    img = Image.open(io.BytesIO(image_data))
    img.save('graph_agents.png')
    display(img)

  except Exception as e:
    print(f"Erro ao processar ou salvar a imagem: {e}")


if __name__ == '__main__':
  agentic()
