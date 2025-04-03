import os
from dotenv import load_dotenv, find_dotenv
import subprocess
from src.utilities.util_functions import join_paths


load_dotenv(find_dotenv())


def python_static_analysis(files):
    outputs = ""
    for file in files:
        command = ["ruff", "check", join_paths(os.getenv("WORK_DIR"), file.filename), "--config", "src/linters/ruff-rules.toml"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if result.stdout.strip() != "All checks passed!":
            outputs += f"\n\n---\n{file.filename}:\n\n{result.stdout}"

    return outputs


# Print results
# print("STDOUT:", result.stdout)
# print("STDERR:", result.stderr)
# print("Exit Code:", result.returncode)

if __name__ == "__main__":
    from src.utilities.objects import CodeFile
    manager_file = CodeFile("src/agents/planer.py")
    file_list = [manager_file]
    print(python_static_analysis(file_list))