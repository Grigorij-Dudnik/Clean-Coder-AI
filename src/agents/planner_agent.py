from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv, find_dotenv
from src.utilities.print_formatters import print_formatted, print_formatted_content_planner
from src.utilities.util_functions import check_file_contents, convert_images, get_joke, read_coderrules, list_directory_tree
from src.utilities.langgraph_common_functions import after_ask_human_condition
from src.utilities.user_input import user_input
from src.utilities.graphics import LoadingAnimation
from src.utilities.llms import init_llms_high_intelligence, init_llms_mini, init_llms_medium_intelligence
import os


load_dotenv(find_dotenv())

llms_planners = init_llms_high_intelligence(run_name="Planner")
llm_strong = llms_planners[0].with_fallbacks(llms_planners[1:])
llms_middle_strength = init_llms_medium_intelligence(run_name="Plan finalizer")
llm_middle_strength = llms_middle_strength[0].with_fallbacks(llms_middle_strength[1:])
llms_controller = init_llms_mini(run_name="Plan Files Controller")
llm_controller = llms_controller[0].with_fallbacks(llms_controller[1:])


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    logic_planner_messages: Sequence[BaseMessage]
    plan_finalizer_messages: Sequence[BaseMessage]
    controller_messages: Sequence[BaseMessage]


parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(f"{parent_dir}/prompts/planner_system.prompt", "r") as f:
    planer_system_prompt_template = f.read()
with open(f"{parent_dir}/prompts/planner_files_controller.prompt", "r") as f:
    files_controller_prompt_template = f.read()
with open(f"{parent_dir}/prompts/planner_logic_pseudocode.prompt", "r") as f:
    logic_planer_system_prompt_template = f.read()
with open(f"{parent_dir}/prompts/planner_finalizer.prompt", "r") as f:
    planer_finalizer_prompt_template = f.read()

animation = LoadingAnimation()

# node functions
def call_simple_planer(state):
    messages = state["messages"]
    print_formatted(get_joke(), color="green")
    animation.start()
    response = llm_strong.invoke(messages)
    animation.stop()
    print_formatted_content_planner(response.content)
    state["messages"].append(response.content)

    # plan_message_for_controller = HumanMessage(content=f"Proposed_plan:\n###\n'''{response.content}'''")
    # controller_response = llm_controller.invoke([state["controller_messages"][0], plan_message_for_controller])
    # print("Plan controller response:")
    # print(controller_response.content)

    ask_human_planner(state)

    return state


def call_advanced_planner(state):
    logic_planner_messages = state["logic_planner_messages"]
    print_formatted(get_joke(), color="green")
    animation.start()
    logic_pseudocode = llm_strong.invoke(logic_planner_messages)
    print_formatted("\nIntermediate planning done. Finalizing plan...", color="light_magenta")
    if os.getenv("SHOW_LOGIC_PLAN"):
        print_formatted(logic_pseudocode.content, color="light_yellow")

    state["plan_finalizer_messages"].append(HumanMessage(content=f"Logic pseudocode plan to follow:\n\n{logic_pseudocode.content}"))
    plan_finalizer_messages = state["plan_finalizer_messages"]
    plan = llm_middle_strength.invoke(plan_finalizer_messages)
    animation.stop()
    print_formatted_content_planner(plan.content)
    state["messages"].append(plan.content)

    ask_human_planner(state)

    return state


def ask_human_planner(state):
    human_message = user_input("Type (o)k if you accept or provide commentary. ")
    if human_message in ['o', 'ok']:
        state["messages"].append(HumanMessage(content="Approved by human"))
    else:
        state["messages"].append(HumanMessage(
            content=f"Plan been rejected by human. Improve it following his commentary: {human_message}"
        ))
    return state


# workflow definition
planner_workflow = StateGraph(AgentState)
planner_workflow.add_node("advanced_planner", call_advanced_planner)
planner_workflow.add_node("agent", call_simple_planer)
planner_workflow.set_entry_point("advanced_planner")
planner_workflow.add_conditional_edges("advanced_planner", after_ask_human_condition)
planner_workflow.add_conditional_edges("agent", after_ask_human_condition)
planner = planner_workflow.compile()


def planning(task, text_files, image_paths, work_dir, documentation=None, dir_tree=None, coderrules=None):
    # that ifs needed for sake of testing (manual tests)
    if not dir_tree:
        dir_tree = list_directory_tree(work_dir)
    if not coderrules:
        coderrules = read_coderrules()
    file_contents = check_file_contents(text_files, work_dir, line_numbers=False)
    basic_planer_system_message = SystemMessage(content=planer_system_prompt_template.format(project_rules=coderrules, file_contents=file_contents, dir_tree=dir_tree))
    logic_planer_system_message = SystemMessage(content=logic_planer_system_prompt_template.format(
        project_rules=coderrules,
        file_contents=file_contents,
        dir_tree=dir_tree
    ))
    planer_finalizer_system_message = SystemMessage(content=planer_finalizer_prompt_template.format(
        project_rules=coderrules,
        file_contents=file_contents,
    ))
    print_formatted("📈 Planner here! Create plan of changes with me!", color="light_blue")
    images = convert_images(image_paths)
    message_content_without_imgs = f"Task:\n'''{task}'''"
    message_without_imgs = HumanMessage(content=message_content_without_imgs)
    controller_system_message = SystemMessage(content=files_controller_prompt_template.format(file_contents=file_contents, dir_tree=dir_tree, task=task))

    inputs = {
        "messages": [basic_planer_system_message, message_without_imgs,],# documentation],
        "logic_planner_messages": [logic_planer_system_message, message_without_imgs,],
        "plan_finalizer_messages": [planer_finalizer_system_message],
        "controller_messages": [controller_system_message]
    }
    if images:
        inputs["messages"].append(HumanMessage(content=images))
        inputs["logic_planner_messages"].append(HumanMessage(content=images))
        inputs["plan_finalizer_messages"].append(HumanMessage(content=images))
    planner_response = planner.invoke(inputs, {"recursion_limit": 50})["messages"][-2]

    return planner_response
