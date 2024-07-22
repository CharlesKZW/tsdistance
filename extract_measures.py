import inspect
import importlib.util
import pandas as pd

# Paths to your files
files = {
    "Elastic Measures Testing Progress Tracking": 'tsdistance/elastic.py',
    "Kernel Measures Testing Progress Tracking": 'tsdistance/kernel.py',
    "Lockstep Measures Testing Progress Tracking": 'tsdistance/lockstep.py',
    "Sliding Measures Testing Progress Tracking": 'tsdistance/sliding.py'
}

# Function to extract measures from a file
def extract_measures(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = [
        {"Measure": name, "Tested": "No", "Testing Method": "", "Notes/Comments": ""}
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) or (hasattr(obj, "__wrapped__") and inspect.isfunction(obj.__wrapped__))
    ]
    return functions

# Create markdown content
markdown_content = ""

for title, file_path in files.items():
    measures = extract_measures(file_path)
    df = pd.DataFrame(measures)
    markdown_content += f"# {title}\n\n"
    markdown_content += df.to_markdown(index=False)
    markdown_content += "\n\n"

# Save to Markdown
with open("DEV_PROGRESS.md", "w") as f:
    f.write(markdown_content)

print("DEV_PROGRESS.md updated successfully.")
