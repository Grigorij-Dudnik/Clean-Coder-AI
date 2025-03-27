"""
File with functions allowing to set up api keys in .env file.

As that functions are set up in the beginning of work process, avoid improrting anything from other files. (Especially from files where env variables are needed).
"""

import os
import sys
from termcolor import colored


def set_up_env_coder_pipeline():
    envs = {}
    print(colored("🖐  Hey! Let's set up our project variables.", color="cyan"))
    print(
        colored(
            "1/3. Provide one or more API keys for LLM providers or the local Ollama model. Don't worry, you can always modify them in the .env file.",
            color="cyan",
        )
    )
    envs["ANTHROPIC_API_KEY"] = input("Provide your Anthropic API key (Optional) (Why? Best coding capabilities):\n")
    envs["OPENAI_API_KEY"] = input(
        "Provide your OpenAI API key (Optional) (Why? Needed for using microphone, best planning capabilities with o1):\n"
    )
    envs["OPEN_ROUTER_API_KEY"] = input("Provide your OpenRouter API key (Optional) (Why? All models in one place):\n")
    print(colored("2/3. Now provide the folder containing your project.", color="cyan"))
    envs["WORK_DIR"] = input("Provide full path to your work directory:\n")
    print(colored("3/3. (Optional) If you want to use frontend feedback feature:", color="cyan"))
    envs["FRONTEND_URL"] = input(
        "Provide url under which your frontend app is running (Optional) (Example: http://localhost:1234):\n"
    )
    # save them to file
    with open(".env", "w") as f:
        for key, value in envs.items():
            f.write(f"{key}={value}\n")

    print(
        colored("We have done .env file set up! You can modify your variables in any moment in .env.\n", color="green")
    )

    if os.getenv("WORK_DIR") == "/work_dir":
        print(colored("Rerun to read variables you just saved.", color="yellow"))
        sys.exit()


def set_up_env_manager():
    envs = {}
    print(colored("🖐  Hey! Let's set up our project variables.", color="cyan"))
    print(
        colored(
            "1/4. Provide one or more API keys for LLM providers or the local Ollama model. Don't worry, you can always modify them in the .env file.",
            color="cyan",
        )
    )
    envs["ANTHROPIC_API_KEY"] = input("Provide your Anthropic API key (Optional) (Why? Best coding capabilities):\n")
    envs["OPENAI_API_KEY"] = input(
        "Provide your OpenAI API key (Optional) (Why? Needed for using microphone, best planning capabilities with o1):\n"
    )
    envs["OPEN_ROUTER_API_KEY"] = input("Provide your OpenRouter API key (Optional) (Why? All models in one place):\n")
    print(colored("2/4. Now provide the folder containing your project.", color="cyan"))
    envs["WORK_DIR"] = input("Provide full path to your work directory:\n")
    print(colored("3/4. (Optional) If you want to use frontend feedback feature:", color="cyan"))
    envs["FRONTEND_URL"] = input(
        "Provide url under which your frontend app is running (Optional) (Example: http://localhost:1234):\n"
    )
    print(colored("4/4. Now let's set up your Todoist connection.", color="cyan"))
    envs["TODOIST_API_KEY"] = input("Please provide your Todoist API key:\n")

    with open(".env", "w") as f:
        for key, value in envs.items():
            f.write(f"{key}={value}\n")

    print(
        colored("We have done .env file set up! You can modify your variables in any moment in .env.\n", color="green")
    )

    if os.getenv("WORK_DIR") == "/work_dir":
        print(colored("Rerun to read variables you just saved.", color="yellow"))
        sys.exit()


def add_todoist_envs():
    envs = {}
    print(colored("1/1. Now let's set up your Todoist connection.", color="cyan"))
    envs["TODOIST_API_KEY"] = input("Provide your Todoist API key:\n")

    with open(".env", "a+") as f:
        for key, value in envs.items():
            f.write(f"{key}={value}\n")

    # load
    for key, value in envs.items():
        if value:  # Only load if the value is not empty
            os.environ[key] = value

    print(
        colored("We have done .env file set up! You can modify your variables in any moment in .env.\n", color="green")
    )
