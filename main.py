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

repl = PythonREPL()
tavily_tool = TavilySearchResults(max_results=5)


class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]
  sender: str


def create_agent(llm, tools, system_message: str):
  """Create an agent."""
  prompt = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        " You have access to the following tools: {tool_names}.\n{system_message}",
      ),
      MessagesPlaceholder(variable_name="messages"),
    ]
  )
  prompt = prompt.partial(system_message=system_message)
  prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
  return prompt | llm.bind_tools(tools)


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
  """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
  try:
    result = repl.run(code)
  except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
  result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
  return (
      result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
  )


def agent_node(state, agent, name):
  result = agent.invoke(state)
  # We convert the agent output into a format that is suitable to append to the global state
  if isinstance(result, ToolMessage):
    pass
  else:
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  return {
    "messages": [result],
    # Since we have a strict workflow, we can
    # track the sender so we know who to pass to next.
    "sender": name,
  }


def router(state) -> Literal["call_tool", "__end__", "continue"]:
  # This is the router
  messages = state["messages"]
  last_message = messages[-1]
  if last_message.tool_calls:
    # The previous agent is invoking a tool
    return "call_tool"
  if "FINAL ANSWER" in last_message.content:
    # Any agent decided the work is done
    return "__end__"
  return "continue"


if __name__ == '__main__':
  # main()
  llm = ChatOpenAI(model="gpt-4-1106-preview")

  # Research agent and node
  research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
  )
  research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

  # chart_generator
  chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
  )
  chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

  tools = [tavily_tool, python_repl]
  tool_node = ToolNode(tools)

  workflow = StateGraph(AgentState)

  workflow.add_node("Researcher", research_node)
  workflow.add_node("chart_generator", chart_node)
  workflow.add_node("call_tool", tool_node)

  workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
  )
  workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
  )

  workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
      "Researcher": "Researcher",
      "chart_generator": "chart_generator",
    },
  )
  workflow.set_entry_point("Researcher")
  graph = workflow.compile()

  events = graph.stream(
    {
      "messages": [
        HumanMessage(
          content="Fetch the UK's GDP over the past 5 years,"
                  " then draw a line graph of it."
                  " Once you code it up, finish."
        )
      ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
  )
  for s in events:
    print(s)
    print("----")
