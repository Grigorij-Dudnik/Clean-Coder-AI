import os
from langchain_core.messages import HumanMessage
from src.utilities.llms import init_llms_medium_intelligence
from src.utilities.start_work_functions import read_frontend_feedback_story
import base64
import textwrap
from typing import Optional
from pydantic import BaseModel, Field


llms = init_llms_medium_intelligence(run_name="Frontend Feedback")

llm = llms[0].with_fallbacks(llms[1:])

# read prompt from file
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(f"{parent_dir}/prompts/frontend_feedback.prompt", "r") as f:
    frontend_feedback_prompt_template = f.read()


class ScreenshotCodingStructure(BaseModel):
    """Output structure"""

    analysis: str = Field(
        description="""
    Think which frontend page needed to be sought in order to provide programmer with valuable feedback (if any).
    """
    )
    questions: Optional[str] = Field(
        default=None,
        description="If you have missing info about some endpoint/selector/other important element name, ask about it here. If everything clear, questions are not needed.",
    )
    screenshot_code: str = Field(
        description="""
    Provide here your playwright code for a screenshot. If screenshot is not needed for that task write exactly "No screenshot needed".
    """
    )


def write_screenshot_codes(task, plan, work_dir):
    story = read_frontend_feedback_story()
    story = story.format(frontend_url=os.environ["FRONTEND_URL"])
    prompt = frontend_feedback_prompt_template.format(
        task=task,
        plan=plan,
        story=story,
    )
    llm_ff = llm.with_structured_output(ScreenshotCodingStructure).with_config({"run_name": "VFeedback"})
    response = llm_ff.invoke(prompt)

    if response.screenshot_code == "No screenshot needed":
        return None

    questions = response.questions
    # fulfill the missing information
    # if questions:
    #     file_answerer = ResearchFileAnswerer(work_dir=work_dir)
    #     answers = file_answerer.research_and_answer(questions)

    playwright_start = """
from playwright.sync_api import sync_playwright

p = sync_playwright().start()
browser = p.chromium.launch(headless=False)
page = browser.new_page()
try:
"""
    playwright_end = """
    output = page.screenshot()
except Exception as e:
    output = f"{type(e).__name__}: {e}"
browser.close()
p.stop()
"""

    indented_playwright_code = textwrap.indent(response.screenshot_code, "    ")
    code = playwright_start + indented_playwright_code + playwright_end
    return code


def execute_screenshot_codes(playwright_code):
    code_execution_variables = {}
    exec(playwright_code, {}, code_execution_variables)
    screenshot_bytes = code_execution_variables["output"]
    if isinstance(screenshot_bytes, str):
        # in case of error instead of screenshot_bytes
        output_message_content = [{"type": "text", "text": screenshot_bytes}]
        return HumanMessage(content=output_message_content, contains_screenshots=True)

    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    output_message_content = [
        {"type": "text", "text": "Screenshot of current app state:"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_base64}",
            },
        },
    ]

    return HumanMessage(content=output_message_content, contains_screenshots=True)
