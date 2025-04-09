import os
from src.tools.tools_coder_pipeline import (
    ask_human_tool,
    prepare_list_dir_tool,
    prepare_see_file_tool,
    prepare_create_file_tool,
    prepare_replace_code_tool,
    prepare_insert_code_tool,
)
from typing import TypedDict, Sequence, List
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from src.utilities.print_formatters import print_formatted
from src.utilities.util_functions import (
    check_file_contents,
    check_application_logs,
    bad_tool_call_looped,
    read_coderrules,
    convert_images,
    list_directory_tree,
)
from src.utilities.script_execution_utils import logs_from_running_script
from src.utilities.llms import init_llms_medium_intelligence
from src.utilities.langgraph_common_functions import (
    call_model,
    call_tool,
    ask_human,
    after_ask_human_condition,
    multiple_tools_msg,
    no_tools_msg,
    agent_looped_human_help,
)
from src.utilities.objects import CodeFile
from src.agents.frontend_feedback import execute_screenshot_codes
from src.linters.static_analisys import python_static_analysis
from src.utilities.util_functions import load_prompt

load_dotenv(find_dotenv())
log_file_path = os.getenv("LOG_FILE")
frontend_url = os.getenv("FRONTEND_URL")
execute_file_name = os.getenv("EXECUTE_FILE_NAME")


@tool
def final_response_debugger(
    test_instruction: Annotated[str, "Detailed instructions for human to test implemented changes"]
):
    """Call that tool when all changes are implemented to tell the job is done."""
    pass


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


system_prompt_template = load_prompt("debugger_system")


class Debugger:
    def __init__(self, files, work_dir, human_feedback, image_paths, playwright_code=None):
        self.work_dir = work_dir
        self.tools = prepare_tools(work_dir)
        self.llms = init_llms_medium_intelligence(self.tools, "Debugger")
        self.system_message = SystemMessage(content=system_prompt_template.format(project_rules=read_coderrules()))
        self.files = files
        self.images = convert_images(image_paths)
        self.human_feedback = human_feedback
        self.playwright_code = playwright_code
        # workflow definition
        debugger_workflow = StateGraph(AgentState)
        debugger_workflow.add_node("agent", self.call_model_debugger)
        debugger_workflow.add_node("check_log", self.check_log)
        debugger_workflow.add_node("frontend_screenshots", self.frontend_screenshots)
        debugger_workflow.add_node("human_help", agent_looped_human_help)
        debugger_workflow.add_node("human_end_process_confirmation", ask_human)

        debugger_workflow.set_entry_point("agent")

        debugger_workflow.add_edge("human_help", "agent")
        debugger_workflow.add_edge("frontend_screenshots", "human_end_process_confirmation")
        debugger_workflow.add_conditional_edges("agent", self.after_agent_condition)
        debugger_workflow.add_conditional_edges("check_log", self.after_check_log_condition)
        debugger_workflow.add_conditional_edges("human_end_process_confirmation", after_ask_human_condition)

        self.debugger = debugger_workflow.compile()

    # node functions
    def call_model_debugger(self, state: dict) -> dict:
        state = call_model(state, self.llms)
        state = call_tool(state, self.tools)
        messages = [msg for msg in state["messages"] if msg.type == "ai"]
        last_ai_message = messages[-1]
        if len(last_ai_message.tool_calls) > 1:
            for tool_call in last_ai_message.tool_calls:
                state["messages"].append(ToolMessage(content="too much tool calls", tool_call_id=tool_call["id"]))
            state["messages"].append(HumanMessage(content=multiple_tools_msg))
        elif len(last_ai_message.tool_calls) == 0:
            state["messages"].append(HumanMessage(content=no_tools_msg))

        for tool_call in last_ai_message.tool_calls:
            if tool_call["name"] == "create_file_with_code":
                filename = tool_call["args"]["filename"]
                new_file = CodeFile(filename, is_modified=True)
                self.files.add(new_file)
                # Dodanie logowania z wersji upstream
                log_message = f"File {filename} created."
                state = write_and_append_log(state, log_message, os.path.join(self.work_dir, "logs.txt"))
            elif tool_call["name"] in ["replace_code", "insert_code"]:
                filename = tool_call["args"]["filename"]
                for file in self.files:
                    if file.filename == filename:
                        file.is_modified = True
                        break
            elif tool_call["name"] == "final_response_debugger":
                files_to_check = [file for file in self.files if file.filename.endswith(".py") and file.is_modified]
                analysis_result = python_static_analysis(files_to_check)
                if analysis_result:
                    state["messages"].append(HumanMessage(content=analysis_result))
                if execute_file_name:
                    message = logs_from_running_script(self.work_dir, execute_file_name, silent_setup=True)
                    state["messages"].append(HumanMessage(content=message))

        return state

    def check_log(self, state: dict) -> dict:
        """Add server logs."""
        logs = check_application_logs()
        log_message = HumanMessage(content="Logs:\n" + logs)
        state["messages"].append(log_message)
        return state

    def frontend_screenshots(self, state):
        print_formatted("Making screenshots, please wait a while...", color="light_blue")
        # Remove old one
        state["messages"] = [msg for msg in state["messages"] if not hasattr(msg, "contains_screenshots")]
        # Add new file contents
        screenshot_msg = execute_screenshot_codes(self.playwright_code)
        state["messages"].append(screenshot_msg)
        return state

    # Conditional edge functions
    def after_agent_condition(self, state):
        messages = [msg for msg in state["messages"] if msg.type in ["ai", "human"]]
        last_message = messages[-1]

        if bad_tool_call_looped(state):
            return "human_help"
        elif hasattr(last_message, "tool_calls") and last_message.tool_calls[0]["name"] == "final_response_debugger":
            if log_file_path:
                return "check_log"
            elif self.playwright_code:
                return "frontend_screenshots"
            else:
                return "human_end_process_confirmation"
        else:
            return "agent"

    def after_check_log_condition(self, state):
        last_message = state["messages"][-1]
        if last_message.content.endswith("Logs are correct"):
            if self.playwright_code:
                return "frontend_screenshots"
            else:
                return "human_end_process_confirmation"
        else:
            return "agent"

    def do_task(self, task: str, plan: str) -> List[CodeFile]:
        print_formatted("Debugger starting its work", color="green")
        print_formatted("🕵️‍♂️ Need to improve your code? I can help!", color="light_blue")
        file_contents = check_file_contents(self.files, self.work_dir)
        inputs = {
            "messages": [
                self.system_message,
                HumanMessage(content=f"Task: {task}\n\n######\n\nPlan which developer implemented already:\n\n{plan}"),
                HumanMessage(content=list_directory_tree(self.work_dir)),
                HumanMessage(content=f"File contents: {file_contents}", contains_file_contents=True),
                HumanMessage(content=f"Human feedback: {self.human_feedback}"),
            ]
        }
        if self.images:
            inputs["messages"].append(HumanMessage(content=self.images))
        if self.playwright_code:
            print_formatted("Making screenshots, please wait a while...", color="light_blue")
            screenshot_msg = execute_screenshot_codes(self.playwright_code)
            inputs["messages"].append(screenshot_msg)
        self.debugger.invoke(inputs, {"recursion_limit": 150})

        return self.files


def prepare_tools(work_dir):
    list_dir = prepare_list_dir_tool(work_dir)
    see_file = prepare_see_file_tool(work_dir)
    replace_code = prepare_replace_code_tool(work_dir)
    insert_code = prepare_insert_code_tool(work_dir)
    create_file = prepare_create_file_tool(work_dir)
    tools = [list_dir, see_file, replace_code, insert_code, create_file, ask_human_tool, final_response_debugger]

    return tools
