import inspect
import importlib.util
import pandas as pd

# Path to your elastic.py file
elastic_file_path = 'tsdistance/elastic.py'

# Load the module from the file path
spec = importlib.util.spec_from_file_location("elastic", elastic_file_path)
elastic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elastic)

# Extract functions from the module
functions = inspect.getmembers(elastic, inspect.isfunction)

# Create a list to store the function details
measures = []

for func_name, func in functions:
    measures.append({
        "Measure": func_name,
        "Tested": "No",
        "Testing Method": "",
        "Notes/Comments": ""
    })

# Convert to DataFrame
df = pd.DataFrame(measures)

# Save to Markdown
with open("DEV_PROGRESS.md", "w") as f:
    f.write("# Distance Measures Testing Progress\n\n")
    f.write("This document tracks the testing progress of various distance measures in our library. It is intended for developers to monitor the testing status.\n\n")
    f.write(df.to_markdown(index=False))

print("DEV_PROGRESS.md updated successfully.")
