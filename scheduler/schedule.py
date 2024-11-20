import os
import subprocess
import textwrap
import inquirer
import re

# Define available GPUs and number of devices
AVAILABLE_GPUS = ['a100', 'v100', 'rtx3080', 'work']

def get_user_input():
    """Function to get user input for scheduling slurm jobs.
    Get the slurm script and experiment to run.
    Also get the GPU and number of devices to run the job on.

    :return: User's custom input
    :rtype: dict
    """
    
    pprint_slurm_scripts = ['BIMAP.slurm']
    
    # Get user input for slurm script to run
    slurm_answers = inquirer.prompt([
        inquirer.List('slurm_script', message="Choose a slurm template", choices=pprint_slurm_scripts)
    ])
    
    pprint_experiments = ['trainer', 'test']
    
    # Get user input for experiment to run, GPU devoce and number of devices
    questions = [
        inquirer.List('experiment', message="Choose an experiment", choices=pprint_experiments, carousel=True),
        inquirer.List('gpu', message="Select a GPU", choices=AVAILABLE_GPUS)
    ]
    
    # Return user input
    return inquirer.prompt(questions) | slurm_answers

def generate_slurm_script(user_input):
    """Function to generate a slurm script based on user input and template.

    :param user_input: User's custom input
    :type user_input: dict
    :return: Path to the generated slurm script
    :rtype: str
    """
    # Get the slurm script path from user input
    slurm_script = os.path.join("slurm", user_input['slurm_script'])

    # Read the slurm script
    with open(slurm_script, "r") as f:
        script_str = f.read()
    
    warning_and_comments = textwrap.dedent("""
    ### ----------------------------------------------------------------------
    ### !! THIS IS A GENERATED SLURM SCRIPT !! DO NOT EDIT !! DO NOT DELETE !!
    ### Lines like "#SBATCH" configure the job resources
    ### (even though they look like bash comments)
    ### ----------------------------------------------------------------------
    """).strip()
    
    script_str = re.sub(r"#!/bin/bash -l", f"#!/bin/bash -l\n\n{warning_and_comments}\n", script_str)
    user_input['num_devices'] = 1
    # Custom syntax for work GPU nodes
    if user_input['gpu'] == 'work':
        script_str = re.sub(r"#SBATCH --partition=\w+\n", "", script_str)
        script_str = re.sub(r"#SBATCH --gres=gpu:(.+)", f"#SBATCH --gres=gpu:{user_input['num_devices']}", script_str)
    else:
        script_str = re.sub(r"#SBATCH --gres=gpu:(.+)", f"#SBATCH --gres=gpu:{user_input['gpu']}:{user_input['num_devices']}", script_str)
        script_str = re.sub(r"#SBATCH --partition=(.+)", f"#SBATCH --partition={user_input['gpu']}", script_str)
    
    # Replace PLACEHOLDER with experiment name from user input
    script_str = script_str.replace("PLACEHOLDER_JOB_NAME", os.path.splitext(user_input['experiment'])[0])
    
    # Generate a temporary slurm script for scheduling
    temp_slurm_script = os.path.join("generated_slurm_script")
    with open(temp_slurm_script, "w") as f:
        f.write(script_str)
    
    return temp_slurm_script


def submit_job(script_path):
    """Function to submit a slurm job using sbatch.

    :param script_path: Path to the slurm script
    :type script_path: str
    :return: Output of the sbatch command
    :rtype: subprocess.CompletedProcess
    """
    cmd = ["sbatch", script_path]
    
    return subprocess.run(cmd)


if __name__ == "__main__":
    user_input = get_user_input()
    script_path = generate_slurm_script(user_input)
    output = submit_job(script_path)
    # print(output)
