import os
import random
import subprocess

# Step 1: Read the commands from setups.txt
filename = 'setups.txt'
with open(filename, 'r') as file:
    commands = file.readlines()

# Randomly select a command
selected_command = random.choice(commands).strip()

# Modify the PATH to use the virtual environment's binaries first
venv_path = os.path.abspath("../venv/bin")  # Adjust as necessary
os.environ["PATH"] = f"{venv_path}:{os.environ['PATH']}"

# Launch the selected command
print(f"Executing: {selected_command}")
subprocess.run(selected_command, shell=True, executable='/bin/bash')