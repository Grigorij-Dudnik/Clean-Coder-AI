from langchain_openai.chat_models import ChatOpenAI as ChatOpenRouter
from langchain_openai.chat_models import ChatOpenAI as ChatLocalModel
from os import getenv
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def llm_open_router(model):
    return ChatOpenRouter(
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model,
        default_headers={
            "HTTP-Referer": "https://github.com/Grigorij-Dudnik/Clean-Coder-AI",
            "X-Title": "Clean Coder",
        },
        timeout=90,
    )


def llm_open_local_hosted(model):
    return ChatLocalModel(
        openai_api_key="n/a",
        openai_api_base=getenv("LOCAL_MODEL_API_BASE"),
        model_name=model,
        timeout=90,
    )


def init_llms_medium_intelligence(tools=None, run_name="Clean Coder", temp=0):
    llms = []
    if getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("anthropic/claude-sonnet-4"))

    if getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))

    if getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="gpt-4.1", temperature=temp, timeout=90))

    for i, llm in enumerate(llms):
        if tools:
            llm = llm.bind_tools(tools)
        llms[i] = llm.with_config({"run_name": run_name})
    return llms


def init_llms_mini(tools=None, run_name="Clean Coder", temp=0):
    llms = []
    if os.getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("anthropic/claude-3.5-haiku"))

    # if os.getenv("GOOGLE_API_KEY"):
    #     llms.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temp, timeout=60))
    if os.getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))

    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="gpt-4.1-mini", temperature=temp, timeout=90))

    for i, llm in enumerate(llms):
        if tools:
            llm = llm.bind_tools(tools)
        llms[i] = llm.with_config({"run_name": run_name})
    return llms


def init_llms_high_intelligence(tools=None, run_name="Clean Coder", temp=0.2):
    llms = []
    if getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("google/gemini-2.5-pro"))
    if getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("anthropic/claude-opus-4"))

    if os.getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))

    if os.getenv("ANTHROPIC_API_KEY"):
        llms.append(ChatAnthropic(model="claude-opus-4-20250514", temperature=temp, timeout=90, max_tokens=8000))
    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="o3", temperature=1, timeout=90, reasoning_effort="high"))

    for i, llm in enumerate(llms):
        if tools:
            llm = llm.bind_tools(tools)
        llms[i] = llm.with_config({"run_name": run_name})
    return llms
