import json
from pathlib import Path

# Load the mapping from JSON
with open("./scripts/new_vs_old_name.json", "r") as json_file:
    name_mapping = json.load(json_file)

# Read the original filenames
fname_case = Path("./cases/test_tumor_grade4.txt")
with open(fname_case, "r") as txt_file:
    filenames = [line.strip() for line in txt_file]

# Map the filenames
updated_filenames = [name_mapping.get(f"{fname[-1]+fname[:-1]}") for fname in filenames]

# Write the updated filenames to a new file
fname_case_new = fname_case.parent / (fname_case.stem + '_npp.txt')
with open(fname_case_new, "w") as output_file:
    for new_name in updated_filenames:
        output_file.write(new_name + "\n")

print("Filename mapping complete. Updated file saved as %s." % fname_case_new)