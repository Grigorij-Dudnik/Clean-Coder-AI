import pathlib
import sys

repo_directory = pathlib.Path(__file__).parents[2].resolve()
sys.path.append(str(repo_directory))
from dotenv import find_dotenv, load_dotenv

from src.agents.executor_agent import Executor
from tests.manual_tests.utils_for_tests import cleanup_work_dir, setup_work_dir

load_dotenv(find_dotenv())

folder_with_project_files = repo_directory.joinpath("tests", "manual_tests", "projects_files", "executor_scenario_1_files")
tmp_folder =  pathlib.Path(__file__).parent.resolve().joinpath("sandbox_work_dir")
setup_work_dir(manual_tests_folder=tmp_folder, test_files_dir=folder_with_project_files)
executor = Executor({"main.py"}, str(tmp_folder))

task = "Create fastapi app with few endpoints."
plan = """1. Create registration_logic.py file:

```python
from db import db
from auth import hash_password


def register_user(email: str, password: str):
    users_collection = db['users']
    
    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(password)
    user_data = {
        "email": email,
        "password": hashed_password,
    }
    interns_collection.insert_one(user_data)
    send_registration_email(email)
    return {"message": "Intern registered successfully"}
```

2. Next, we need to create endpoint in main.py for handling registration:

first, add import statement on the beginning:
```python
+ from registration_logic import register_user
```

now, create endpoint:
```python
@app.post('/register')
async def register_endpoint():
    data = await request.json()
    email = data.get("email")
    password = data.get("password")

    return register_user(email, password)
```

That's it!

"""

executor.do_task(task, plan)
cleanup_work_dir(manual_tests_folder=tmp_folder)
