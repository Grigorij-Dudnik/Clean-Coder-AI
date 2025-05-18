from langchain_core.messages import HumanMessage
from src.utilities.print_formatters import (
    print_formatted,
    print_formatted_content,
)
from src.utilities.util_functions import invoke_tool_native, TOOL_NOT_EXECUTED_WORD
from src.utilities.user_input import user_input
from langgraph.graph import END
from src.utilities.graphics import LoadingAnimation
import sys


multiple_tools_msg = (
    TOOL_NOT_EXECUTED_WORD
    + """You made multiple tool calls at once. If you want to execute 
multiple actions, choose only one for now; rest you can execute later."""
)
no_tools_msg = (
    TOOL_NOT_EXECUTED_WORD
    + """Please provide a tool call to execute an action. If all needed actions are done, you can call 'final_response' tool."""
)
empty_message_msg = TOOL_NOT_EXECUTED_WORD + "Empty messages are not allowed."
finish_too_early_msg = (
    TOOL_NOT_EXECUTED_WORD
    + """You want to call final response with other tool calls. Don't you finishing too early?"""
)


animation = LoadingAnimation()


# nodes
def _get_llm_response(llms, messages, printing):
    for llm in llms:
        try:
            return llm.invoke(messages)
        except Exception as e:
            if printing:
                print_formatted(
                    f"\nException happened: {e} with llm: {llm.bound.__class__.__name__}. "
                    "Switching to next LLM if available...",
                    color="yellow",
                )
    if printing:
        print_formatted("Can not receive response from any llm", color="red")
    sys.exit()


def call_model(state, llms, printing=True):
    messages = state["messages"]

    if printing:
        animation.start()
    response = _get_llm_response(llms, messages, printing)
    if printing:
        animation.stop()

    if printing:
        print_formatted_content(response)
    state["messages"].append(response)

    return state


def call_tool(state, tools):
    """
    Execute tool calls in a safe order:
    1. Line-based modifications (`start_line` provided) are executed from the
       greatest line number to the smallest, preventing index shifts.
    2. All other calls are executed afterwards, preserving the model’s order.
    """
    last_message = state["messages"][-1]

    ordered_calls = _sort_tool_calls(last_message.tool_calls)

    tool_response_messages = [
        invoke_tool_native(tool_call, tools) for tool_call in ordered_calls
    ]
    state["messages"].extend(tool_response_messages)
    return state


def _sort_tool_calls(tool_calls):
    """
    Return list of tool calls where calls containing `start_line`
    are sorted descending by that value, followed by all remaining calls.
    """
    calls_with_start_line = [
        call for call in tool_calls if "start_line" in call.get("args", {})
    ]
    other_calls = [
        call for call in tool_calls if "start_line" not in call.get("args", {})
    ]

    # Largest start_line first
    calls_with_start_line.sort(
        key=lambda call: call["args"]["start_line"], reverse=True
    )

    return calls_with_start_line + other_calls


def ask_human(state):
    human_message = user_input("Type (o)k to accept or provide commentary. ")
    if human_message in ["o", "ok"]:
        state["messages"].append(HumanMessage(content="Approved by human"))
    else:
        state["messages"].append(HumanMessage(content=human_message))
    return state


def agent_looped_human_help(state):
    human_message = user_input(
        "It seems the agent repeatedly tries to introduce wrong changes. Help him to find his mistakes."
    )
    state["messages"].append(HumanMessage(content=human_message))
    return state


# conditions
def after_ask_human_condition(state):
    last_message = state["messages"][-1]

    if last_message.content == "Approved by human":
        return END
    else:
        return "agent"
