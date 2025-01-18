import pathlib
import sys

repo_directory = pathlib.Path(__file__).parents[2].resolve()
sys.path.append(str(repo_directory))
from src.agents.executor_agent import Executor
from tests.manual_tests.utils_for_tests import cleanup_work_dir

tmp_folder =  pathlib.Path(__file__).parent.resolve().joinpath("sandbox_work_dir")
executor = Executor(set(), str(tmp_folder))

task = "Create fastapi app with few endpoints."
plan = """1. Create main.py in flask looking like:
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask application!"

@app.route('/hello/<name>')
def hello_name(name):
    return "Hello " + name

@app.route('/square/<int:number>')
def square(number):
    return jsonify(result=number**2)

if __name__ == '__main__':
    app.run(debug=True)
```
"""

executor.do_task(task, plan)
cleanup_work_dir(manual_tests_folder=tmp_folder)
