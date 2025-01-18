import pathlib
import sys

repo_directory = pathlib.Path(__file__).parents[2].resolve()
sys.path.append(str(repo_directory))

from dotenv import find_dotenv, load_dotenv

from src.agents.planner_agent import planning
from tests.manual_tests.utils_for_tests import cleanup_work_dir, get_filenames_in_folder, setup_work_dir

load_dotenv(find_dotenv())

folder_with_project_files = repo_directory.joinpath("tests", "manual_tests", "projects_files", "planner_scenario_2_files")
tmp_folder =  pathlib.Path(__file__).parent.resolve().joinpath("sandbox_work_dir")
setup_work_dir(manual_tests_folder=tmp_folder, test_files_dir=folder_with_project_files)

task = """
Add Matches tab to main page
Definition of done:
1. Update main page navigation:
   - Add "Matches" tab to existing navigation in app/page.tsx:
     ```jsx
     {{["Explore", "Received", "Sent", "Matches"].map((tab) => (
       <button
         key={{tab}}
         onClick={{() => handleTabClick(tab)}}
         className={{`flex-1 px-6 py-2.5 rounded-full text-sm transition-all duration-300 ${{
           activeTab === tab
             ? "bg-white font-medium text-black shadow-sm text-[15px]"
             : "text-gray-400/80 hover:text-gray-500"
         }}`}}
       >
         {{tab}}
       </button>
     ))}}
     ```
   - Update TypeScript types for tab states
   - Add proper styling for 4 tabs layout

2. Create backend endpoint for matches:
   - Create new GET /matches endpoint
   - Return all profiles where invitation status is "accepted"
   - Include all necessary profile information
   - Add proper error handling

3. Implement matches list functionality:
   - Add matches state to main page:
     ```typescript
     const [matchedItems, setMatchedItems] = useState<ProfileItem[]>([]);
     ```
   - Create fetchMatches function to get data from /matches endpoint
   - Add proper error handling
   - Add loading states

4. Update UI for matches tab:
   - Use existing ProfileCard component with match-specific actions
   - Show matched profiles with their information
   - Add empty state for no matches
   - Add proper loading states
   - Add error handling with user feedback

5. Update main page state management:
   - Add matches to listToRender logic
   - Update handleTabClick to fetch matches
   - Ensure proper data refresh after invitation actions

Resources:
- Current main page implementation (app/page.tsx)
- Current ProfileCard component
- Existing tab navigation code
- Existing API integration patterns

Technical notes:
- Follow existing tab implementation pattern
- Use proper TypeScript types
- Add proper loading states
- Implement error handling
- Ensure mobile responsiveness
- Update UI to match design specifications
- Consider pagination for large lists
- Add proper logging for debugging
- Update all related interfaces and types,
"""

files = get_filenames_in_folder(manual_tests_folder=tmp_folder)

directory_tree = """
📁 app
│ ├── .coderrules
📁 Backend
│ ├── auth.py
│ ├── azure_blob_storage.py
│ ├── db.py
│ ├── email_service.py
│ ├── invitations.py
│ ├── main.py
│ ├── profile.py
│ ├── registration.py
│ ├── requirements.txt
│ └── __init__.py
│ └──📁 tests
│ │ └── test_1.py
📁 frontend
│ ├── .env.development
│ ├── .env.production
│ ├── next-env.d.ts
│ ├── next.config.ts
│ ├── package-lock.json
│ ├── package.json
│ ├── postcss.config.mjs
│ ├── README.md
│ ├── tailwind.config.ts
│ └── tsconfig.json
│ └──📁 app
│ │ ├── favicon.ico
│ │ ├── globals.css
│ │ ├── layout.tsx
│ │ └── page.tsx
│ │ └──📁 components
│ │ │ ├── PopupNotification.tsx
│ │ │ └── ProfileCard.tsx
│ │ └──📁 forgot-password
│ │ │ └── page.tsx
│ │ └──📁 login
│ │ │ ├── LoginForm.tsx
│ │ │ └── page.tsx
│ │ └──📁 register
│ │ │ ├── page.tsx
│ │ │ ├── RegistrationCompletedStep.tsx
│ │ │ ├── Step1.tsx
│ │ │ ├── Step2.tsx
│ │ │ └── Step3.tsx
│ │ └──📁 reset-password
│ │ │ └──📁 [token]
│ │ │ │ └── page.tsx
│ │ └──📁 styles
│ │ │ ├── ContentWrapper.tsx
│ │ │ └── uiElements.tsx
│ │ └──📁 survey
│ │ │ ├── page.tsx
│ │ │ └── surveyCategories.ts
│ │ │ └──📁 finish
│ │ │ │ └── page.tsx
│ │ │ └──📁 step
│ │ │ │ └──📁 [stepId]
│ │ │ │ │ └── page.tsx
│ └──📁 public
│ │ ├── award_star.svg
│ │ └── profile.svg
│ │ └──📁 designs
│ │ │ ├── basic_templates.png
│ │ │ ├── Log in.png
│ │ │ ├── main_page-Desktop.png
│ │ │ ├── Main_page-mobile.png
│ │ │ ├── Profile_with_notification.png
│ │ │ ├── → Forgot password_(1).png
│ │ │ └── → Forgot password_.png
"""
coderrules = """
My app is a test app - sort of dating app.
Comment your code to make it understandable for reader.

Frontend:
Frontend is created in NextJS + Tailwind. Ensure to use only the last version of NextJS syntax (13+). Use file-based routing.
Common styles should be placed in app/styles/.
uiElements.tsx is file with common elements as buttons, dropdowns etc - use it always when intoducing visual changes.

When when styling any component, try to use as less styles as possible. Readability is a priority.

When making requests to backend, backend url saved under NEXT_PUBLIC_API_URL env variable.

Backend:
FastApi.
- Always write imports on top of the file.
- Separate functions by 3 newlines.
Place endpoints on main.py only, but advanced logic inside of endpoints keep in functions in different files."""

planning(task, files, image_paths={}, work_dir=str(tmp_folder), dir_tree=directory_tree, coderrules=coderrules)
cleanup_work_dir(manual_tests_folder=tmp_folder)
