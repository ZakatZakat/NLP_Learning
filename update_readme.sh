#!/bin/bash

# Convert the Jupyter Notebook to a Python script and run it
jupyter nbconvert --to script ./NLP/Notebooks/classification_problem/notebook.ipynb --output generate_data_script

# Run the converted script and capture the output
OUTPUT=$(python3 generate_data_script.py)

# Update the README file
echo "## Results" > README.md
echo "\`\`\`" >> README.md
echo "$OUTPUT" >> README.md
echo "\`\`\`" >> README.md

# Optionally, add more static or dynamic content to the README
echo "More details about the project. UPD" >> README.md