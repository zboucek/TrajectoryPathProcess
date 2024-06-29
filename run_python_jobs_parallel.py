import multiprocessing
import subprocess
import os, sys
import glob
import argparse

# Set up argparse to get parameters   
parser = argparse.ArgumentParser()
parser.add_argument("--world_name", type=str, default='0')
args = parser.parse_args()

def run_script(configs):
    """    Run a Python script with the given configurations.   """
    current_folder = os.path.dirname(sys.argv[0])
    script_path = os.path.join(current_folder,"run_python_jobs.py")
    subprocess.call(["python", script_path, "--config", configs])

# set paths and get config list
current_folder = os.path.dirname(sys.argv[0])
config_folder = os.path.join(current_folder, "configs")
configs = glob.glob(config_folder + "/config_"+args.world_name+"_*.json")
# run jobs
script = os.path.join(current_folder,"run_python_jobs.py")
pool = multiprocessing.Pool(processes=len(configs))  
pool.map(run_script, configs)