"""Utilities for loading LLMs."""
import os
from collections.abc import Callable
from os import getenv

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI as ChatLocalModel
from langchain_openai.chat_models import ChatOpenAI as ChatOpenRouter

#from langchain_google_genai import ChatGoogleGenerativeAI

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
    timeout=60,
)

def llm_open_local_hosted(model: str) -> Callable:
    """Return a locally hosted model."""
    return ChatLocalModel(
    openai_api_key="n/a",
    openai_api_base=getenv("LOCAL_MODEL_API_BASE"),
    model_name=model,
    timeout=90,
)

def llms_with_tools_and_config(llms: list[Callable], tools: list[Callable] | None, run_name: str) -> list[Callable]:
    """Adds tools and config to loaded llms."""
    for i, llm in enumerate(llms):
        if tools:
            llm = llm.bind_tools(tools)
        llms[i] = llm.with_config({"run_name": run_name})
    return llms


def init_llms(tools: None | list[Callable] =None, run_name: str = "Clean Coder", temp: float = 0) -> list[Callable]:
    """Returns available mid-sized LLM models, with tools when available and config."""
    llms = []
    if getenv("ANTHROPIC_API_KEY"):
        llms.append(ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temp, timeout=60, max_tokens=2048))
    if getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("anthropic/claude-3.5-sonnet"))
    if getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="gpt-4o", temperature=temp, timeout=60))
    # if os.getenv("GOOGLE_API_KEY"):
    #     llms.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temp, timeout=60))
    if getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE") and getenv("LOCAL_MODEL_NAME"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))
    return llms_with_tools_and_config(llms=llms, tools=tools, run_name=run_name)


def init_llms_mini(tools: None | list[Callable] =None, run_name: str = "Clean Coder", temp: float=0) -> list[Callable]:
    """Returns available small LLM models, with tools when available and config."""
    llms = []
    if os.getenv("ANTHROPIC_API_KEY"):
        llms.append(ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=temp, timeout=60))
    if os.getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("anthropic/claude-3.5-haiku"))
    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="gpt-4o-mini", temperature=temp, timeout=60))
    # if os.getenv("GOOGLE_API_KEY"):
    #     llms.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temp, timeout=60))
    if os.getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE") and getenv("LOCAL_MODEL_NAME"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))
    return llms_with_tools_and_config(llms=llms, tools=tools, run_name=run_name)


def init_llms_high_intelligence(tools: None | list[Callable] =None, run_name: str = "Clean Coder", temp: float=0.2) -> list[Callable]:
    """Returns available high power LLM models, with tools when available and config."""
    llms = []
    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="o3-mini", temperature=1, timeout=60, reasoning_effort="high"))
    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="o1", temperature=1, timeout=60))
    if os.getenv("OPENROUTER_API_KEY"):
        llms.append(llm_open_router("openai/gpt-4o"))
    if os.getenv("OPENAI_API_KEY"):
        llms.append(ChatOpenAI(model="gpt-4o", temperature=temp, timeout=60))
    if os.getenv("ANTHROPIC_API_KEY"):
        llms.append(ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=temp, timeout=60, max_tokens=2048))
    # if os.getenv("GOOGLE_API_KEY"):
    #     llms.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temp, timeout=60))
    if os.getenv("OLLAMA_MODEL"):
        llms.append(ChatOllama(model=os.getenv("OLLAMA_MODEL")))
    if getenv("LOCAL_MODEL_API_BASE") and getenv("LOCAL_MODEL_NAME"):
        llms.append(llm_open_local_hosted(getenv("LOCAL_MODEL_NAME")))
    return llms_with_tools_and_config(llms=llms, tools=tools, run_name=run_name)
