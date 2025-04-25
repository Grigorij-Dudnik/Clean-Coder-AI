"""
This file contains a manual test scenario for the Manager agent.
It uses a dummy finish_project_planning tool to simulate task completion without running the actual coding pipeline.
"""

from src.tools.tools_project_manager import add_task, modify_task, reorder_tasks
from src.tools.tools_coder_pipeline import prepare_list_dir_tool, prepare_see_file_tool, ask_human_tool
from manager import Manager
from langchain.tools import tool
from typing_extensions import Annotated
from src.utilities.print_formatters import print_formatted

@tool
def dummy_finish_project_planning(dummy: Annotated[str, "Type 'ok' to proceed."]):
    """Dummy version of finish_project_planning that simulates task completion without running the actual pipeline."""
    print_formatted("🔄 Simulating task completion...", color="yellow")
    return "Task completed successfully"

class TestManager(Manager):
    def prepare_tools(self):
        """Override prepare_tools to use dummy finish_project_planning"""
        list_dir = prepare_list_dir_tool(self.work_dir)
        see_file = prepare_see_file_tool(self.work_dir)
        tools = [
            add_task,
            modify_task,
            reorder_tasks,
            ask_human_tool,
            dummy_finish_project_planning,
            list_dir,
            see_file,
        ]
        return tools

if __name__ == "__main__":
    print_formatted("🧪 Starting Manager Test Scenario 1", color="green")
    test_manager = TestManager()
    test_manager.run()
    print_formatted("✅ Test scenario completed", color="green")