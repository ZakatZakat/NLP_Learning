#!/bin/bash

# Run the Python script and capture output
OUTPUT=$(python3 generate_data.py)

# Update the README file
echo "## Results" > README.md
echo "\`\`\`" >> README.md
echo "$OUTPUT" >> README.md
echo "\`\`\`" >> README.md

# Optionally, add more static or dynamic content to the README
echo "More details about the project." >> README.md
