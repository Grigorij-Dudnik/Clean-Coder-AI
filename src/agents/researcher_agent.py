from typing import TypedDict, Sequence, List
from src.utilities.objects import CodeFile
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from src.tools.tools_coder_pipeline import (
    prepare_see_file_tool,
    prepare_list_dir_tool,
    retrieve_files_by_semantic_query,
)
from src.tools.rag.retrieval import vdb_available
from src.utilities.util_functions import list_directory_tree, read_coderrules, load_prompt
from src.utilities.langgraph_common_functions import (
    call_model,
    call_tool,
    ask_human,
    after_ask_human_condition,
    no_tools_msg,
)
from src.utilities.print_formatters import print_formatted
from src.utilities.llms import init_llms_medium_intelligence
import os


load_dotenv(find_dotenv())
mistral_api_key = os.getenv("MISTRAL_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
work_dir = os.getenv("WORK_DIR")


@tool
def final_response_researcher(
    files_to_work_on: Annotated[List[str], "List of existing files to potentially introduce changes"],
    reference_files: Annotated[
        List[str],
        "List of code files useful as a reference. There are files where similar task been implemented already.",
    ],
    template_images: Annotated[List[str], "List of template images"],
):
    """That tool outputs list of files programmer will need to change and paths to graphical patterns if some.
    Use that tool only when you 100% sure you found all the files programmer will need to modify.
    If not, do additional research. Include only the files you convinced will be useful.
    Provide only existing files, do not provide files to be implemented.
    """
    pass


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


class Researcher:
    def __init__(self, work_dir, silent=False):
        self.silent = silent
        see_file = prepare_see_file_tool(work_dir)
        list_dir = prepare_list_dir_tool(work_dir)
        self.tools = [see_file, list_dir, final_response_researcher]
        if vdb_available():
            self.tools.append(retrieve_files_by_semantic_query)
        self.llms = init_llms_medium_intelligence(self.tools, "Researcher")

        # workflow definition
        researcher_workflow = StateGraph(AgentState)

        researcher_workflow.add_node("agent", self.call_model_researcher)
        researcher_workflow.add_node("human", ask_human)

        researcher_workflow.set_entry_point("agent")

        researcher_workflow.add_conditional_edges("agent", self.after_agent_condition)
        researcher_workflow.add_conditional_edges("human", after_ask_human_condition)

        self.researcher = researcher_workflow.compile()

    # node functions
    def call_model_researcher(self, state):
        state = call_model(state, self.llms, printing=not self.silent)
        last_message = state["messages"][-1]
        if len(last_message.tool_calls) == 0:
            state["messages"].append(HumanMessage(content=no_tools_msg))
            return state
        elif len(last_message.tool_calls) > 1:
            # Filter out the tool call with "final_response_researcher"
            state["messages"][-1].tool_calls = [
                tool_call for tool_call in last_message.tool_calls if tool_call["name"] != "final_response_researcher"
            ]
        state = call_tool(state, self.tools)
        return state

    # condition functions
    def after_agent_condition(self, state):
        messages = [msg for msg in state["messages"] if msg.type in ["ai", "human"]]
        last_message = messages[-1]

        if last_message.content == no_tools_msg:
            return "agent"
        elif last_message.tool_calls[0]["name"] == "final_response_researcher":
            if self.silent:
                state["messages"].append(HumanMessage(content="Approved automatically"))    # Dummy message to fullfil state, to align with "Approved by human" message in loun mode
                return END  # Skip human approval in silent mode
            return "human"
        else:
            return "agent"

    # just functions
    def research_task(self, task):
        if not self.silent:
            print_formatted("Researcher starting its work", color="green")
            print_formatted("👋 Hey! I'm looking for files on which we will work on together!", color="light_blue")

        system_prompt_template = load_prompt("researcher_system")
        system_message = system_prompt_template.format(task=task, project_rules=read_coderrules())
        inputs = {
            "messages": [SystemMessage(content=system_message), HumanMessage(content=list_directory_tree(work_dir))]
        }
        researcher_response = self.researcher.invoke(inputs, {"recursion_limit": 100})["messages"][-3]
        response_args = researcher_response.tool_calls[0]["args"]
        text_files = set(CodeFile(f) for f in response_args["files_to_work_on"] + response_args["reference_files"])
        image_paths = response_args["template_images"]

        return text_files, image_paths


if __name__ == "__main__":
    task = """Check all system"""
    researcher = Researcher(work_dir)
    researcher.research_task(task)
