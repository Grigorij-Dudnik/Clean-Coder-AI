from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from src.tools.tools_coder_pipeline import (
     prepare_see_file_tool, prepare_list_dir_tool, retrieve_files_by_semantic_query
)
from src.tools.rag.retrieval import vdb_available
from src.utilities.util_functions import list_directory_tree, read_coderrules
from src.utilities.langgraph_common_functions import (
    call_model, call_tool, ask_human, after_ask_human_condition, no_tools_msg
)
from src.utilities.print_formatters import print_formatted
from src.utilities.llms import init_llms_mini
import os


load_dotenv(find_dotenv())
mistral_api_key = os.getenv("MISTRAL_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
work_dir = os.getenv("WORK_DIR")


@tool
def final_response_researcher(files_to_work_on, reference_files, template_images):
    """That tool outputs list of files programmer will need to change and paths to graphical patterns if some.
    Use that tool only when you 100% sure you found all the files programmer will need to modify.
    If not, do additional research. Include only the files you convinced will be useful.
    Provide only existing files, do not provide files to be implemented.

    tool input:
    :param files_to_work_on: ["List", "of", "existing files", "to potentially introduce", "changes"],
    :param reference_files: ["List", "of code files", "useful as a reference", "without images"],
    :param template_images: ["List of", "template", "images"],
    """
    pass

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(f"{parent_dir}/prompts/researcher_system.prompt", "r") as f:
    system_prompt_template = f.read()


# Logic for conditional edges
def after_agent_condition(state):
    messages = [msg for msg in state["messages"] if msg.type in ["ai", "human"]]
    last_message = messages[-1]

    if last_message.content == no_tools_msg:
        return "agent"
    elif last_message.tool_calls[0]["name"] == "final_response_researcher":
        return "human"
    else:
        return "agent"


class Researcher():
    def __init__(self, work_dir):
        see_file = prepare_see_file_tool(work_dir)
        list_dir = prepare_list_dir_tool(work_dir)
        self.tools = [see_file, list_dir, final_response_researcher]
        if vdb_available():
            self.tools.append(retrieve_files_by_semantic_query)
        self.llms = init_llms_mini(self.tools, "Researcher")

        # workflow definition
        researcher_workflow = StateGraph(AgentState)

        researcher_workflow.add_node("agent", self.call_model_researcher)
        researcher_workflow.add_node("human", ask_human)

        researcher_workflow.set_entry_point("agent")

        researcher_workflow.add_conditional_edges("agent", after_agent_condition)
        researcher_workflow.add_conditional_edges("human", after_ask_human_condition)

        self.researcher = researcher_workflow.compile()

    # node functions
    def call_model_researcher(self, state):
        state = call_model(state, self.llms)
        last_message = state["messages"][-1]
        if len(last_message.tool_calls) == 0:
            state["messages"].append(HumanMessage(content=no_tools_msg))
            return state
        elif len(last_message.tool_calls) > 1:
            # Filter out the tool call with "final_response_researcher"
            state["messages"][-1].tool_calls = [
                tool_call for tool_call in last_message.tool_calls
                if tool_call["name"] != "final_response_researcher"
            ]
        state = call_tool(state, self.tools)
        return state

    # just functions
    def research_task(self, task):
        print_formatted("Researcher starting its work", color="green")
        print_formatted("👋 Hey! I'm looking for files on which we will work on together!", color="light_blue")

        system_message = system_prompt_template.format(task=task, project_rules=read_coderrules())
        inputs = {
            "messages": [SystemMessage(content=system_message), HumanMessage(content=list_directory_tree(work_dir))]}
        researcher_response = self.researcher.invoke(inputs, {"recursion_limit": 100})["messages"][-3]
        response_args = researcher_response.tool_calls[0]["args"]
        text_files = set(response_args["files_to_work_on"] + response_args["reference_files"])
        image_paths = response_args["template_images"]

        return text_files, image_paths


if __name__ == "__main__":
    task = """Check all system"""
    researcher = Researcher(work_dir)
    researcher.research_task(task)
