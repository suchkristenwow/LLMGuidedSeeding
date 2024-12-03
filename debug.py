from LLMGuidedSeeding_pkg.utils.gen_utils import dictify  
import json 

response = """
Based on the information provided and the image displaying a concrete pathway adjacent to a grassy area, it appears the seeding mechanism encountered a problem due to attempting to plant in an impenetrable surface — essentially, the concrete part of the pathway.

To avoid such issues in future tasks, we need to update the constraint dictionary to specify avoidance of surfaces where planting is not possible, such as concrete or other hard surfaces. This can be achieved by modifying the "avoid" key in the constraints dictionary to include "impenetrable surfaces".

Here is the revised constraint dictionary including the previous necessary constraints and with the new specification:

```
{
  "meta-obstacle": null,
  "avoid": ["impenetrable surfaces"],  # Avoid concrete and other hard surfaces that prevent planting.
  "goal_lms": "field flag",
  "pattern": null,
  "landmark_offset": 0.05,  # Maintain a 5 cm distance from the landmark (field flag) for seeding.
  "search": null,
  "seed": True,  # Activate the seeding mechanism to plant seeds.
  "pattern_offset": null
}
```

This updated constraint configuration ensures the robot can map out and avoid areas unsuitable for planting (like concrete areas or other similarly hard, unyielding surfaces). This should prevent the exertion of excessive force by the seeding mechanism and avoid potential damages or planting failures in future tasks. Likewise, the description under "avoid" helps the robot's perception system to identify these areas and reroute accordingly.
"""
def clean_json_string(json_string):
    """
    Cleans a JSON-like string by removing comments and ensuring proper formatting.
    Ensures commas are correctly placed between key-value pairs without adding trailing commas.
    """
    lines = json_string.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.split("#")[0].split("//")[0].strip()  # Remove comments
        if line:
            cleaned_lines.append(line)

    # Join lines into a single string
    cleaned_json = "\n".join(cleaned_lines)

    # Replace Python-style True/False/None with JSON-compliant true/false/null
    cleaned_json = cleaned_json.replace("True", "true").replace("False", "false").replace("None", "null")

    # Remove any trailing commas inside JSON objects or arrays
    cleaned_json = cleaned_json.replace(",\n}", "\n}")
    cleaned_json = cleaned_json.replace(",\n]", "\n]")

    return cleaned_json


i0 = response.index("{"); i1 = response.index("}")
parsed_results = response[i0:i1+1]
if "}" not in parsed_results:
    parsed_results = parsed_results + "}"

cleaned_json = clean_json_string(parsed_results)
print("Cleaned JSON:", cleaned_json)

# Convert the cleaned JSON string into a dictionary
try:
    new_constraints = json.loads(cleaned_json)
    print("Parsed Constraints:", new_constraints)
except json.JSONDecodeError as e:
    print("Error parsing JSON:", str(e))