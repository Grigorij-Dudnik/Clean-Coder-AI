from src.tools.tools_coder_pipeline import (
    ask_human_tool,
    prepare_create_file_tool,
    prepare_replace_code_tool,
    prepare_insert_code_tool,
)
from typing import TypedDict, Sequence, List
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from src.utilities.llms import init_llms_medium_intelligence
from src.utilities.print_formatters import print_formatted
from src.utilities.util_functions import check_file_contents, exchange_file_contents, bad_tool_call_looped, load_prompt, TOOL_NOT_EXECUTED_WORD
from src.utilities.langgraph_common_functions import (
    call_model,
    call_tool,
    multiple_tools_msg,
    no_tools_msg,
    agent_looped_human_help,
)
from src.utilities.objects import CodeFile


load_dotenv(find_dotenv())

@tool
def final_response_executor(
    test_instruction: Annotated[str, "Detailed instructions for human to test implemented changes"]
):
    """Call that tool when all plan steps are implemented to finish your job."""
    pass


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


system_prompt_template = load_prompt("executor_system")

class Executor:
    def __init__(self, files, work_dir):
        self.work_dir = work_dir
        self.tools = prepare_tools(work_dir)
        self.llms = init_llms_medium_intelligence(self.tools, "Executor")
        self.system_message = SystemMessage(content=system_prompt_template)
        self.files = files

        # workflow definition
        executor_workflow = StateGraph(AgentState)

        executor_workflow.add_node("agent", self.call_model_executor)
        executor_workflow.add_node("human_help", agent_looped_human_help)

        executor_workflow.set_entry_point("agent")

        executor_workflow.add_edge("human_help", "agent")
        executor_workflow.add_conditional_edges("agent", self.after_agent_condition)

        self.executor = executor_workflow.compile()

    # node functions
    def call_model_executor(self, state):
        """
        Calls of Executor's LLM and after receiving its response calls tools. Next performs alluaxury actions
        depending on last message from LLM. After it exchanges contents of files in agent's context to provide it with
        updated version after inserting changes into file.
        """
        state = call_model(state, self.llms)
        state = call_tool(state, self.tools)

        # auxiliary actions depending on tools called
        ai_messages = [msg for msg in state["messages"] if msg.type == "ai"]
        last_ai_message = ai_messages[-1]
        # if len(last_ai_message.tool_calls) > 1:
        #     for tool_call in last_ai_message.tool_calls:
        #         state["messages"].append(ToolMessage(content="too much tool calls", tool_call_id=tool_call["id"]))
        #     state["messages"].append(HumanMessage(content=multiple_tools_msg))
        if len(last_ai_message.tool_calls) == 0:
            state["messages"].append(HumanMessage(content=no_tools_msg))

        for tool_call in last_ai_message.tool_calls:
            if tool_call["name"] == "create_file_with_code":
                new_file = CodeFile(tool_call["args"]["filename"], is_modified=True)
                self.files.add(new_file)
            elif tool_call["name"] in ["replace_code", "insert_code"]:
                last_tool_message = [msg for msg in state["messages"] if msg.type == "tool"][-1]
                # do not mark as modified if tool was not executed
                if last_tool_message.content.startswith(TOOL_NOT_EXECUTED_WORD):
                    continue
                filename = tool_call["args"]["filename"]
                for file in self.files:
                    if file.filename == filename:
                        file.is_modified = True
                        break

        state = exchange_file_contents(state, self.files, self.work_dir)

        return state

    # Conditional edge functions
    def after_agent_condition(self, state):
        messages = [msg for msg in state["messages"] if msg.type in ["ai", "human"]]
        last_message = messages[-1]

        if bad_tool_call_looped(state):
            return "human_help"
        # final response case
        elif (
            hasattr(last_message, "tool_calls")
            and len(last_message.tool_calls) > 0
            and last_message.tool_calls[0]["name"] == "final_response_executor"
        ):
            return END
        else:
            return "agent"

    # just functions
    def do_task(self, task: str, plan: str) -> List[CodeFile]:
        print_formatted("Executor starting its work", color="green")
        print_formatted("✅ I follow the plan and will implement necessary changes!", color="light_blue")
        file_contents = check_file_contents(self.files, self.work_dir)
        inputs = {
            "messages": [
                self.system_message,
                HumanMessage(content=f"Task: {task}\n\n######\n\nPlan:\n\n{plan}"),
                HumanMessage(content=f"File contents: {file_contents}", contains_file_contents=True),
            ]
        }
        self.executor.invoke(inputs, {"recursion_limit": 150})

        return self.files


def prepare_tools(work_dir):
    replace_code = prepare_replace_code_tool(work_dir)
    insert_code = prepare_insert_code_tool(work_dir)
    create_file = prepare_create_file_tool(work_dir)
    tools = [replace_code, insert_code, create_file, ask_human_tool, final_response_executor]

    return tools
