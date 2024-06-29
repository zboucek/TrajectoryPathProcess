# Path Planning and Trajectory Optimization for Drones

This project is a collection of Python scripts and Jupyter Notebooks for path planning and trajectory optimization for drones in 2D and 3D environments. The project includes implementations of A* and LT* algorithms for path planning and a pseudospectral method for trajectory optimization.

## Installation
To use the project, you will need to install the following Python packages:

- numpy
- matplotlib
- scipy
- cloudpickle
- pyomo

You can install these packages using pip:

`pip install numpy matplotlib scipy cloudpickle pyomo`

You will also need to install Jupyter Notebook to run the notebook files. You can install Jupyter Notebook using pip:

`pip install jupyter`

## Usage
The project includes the following files:

- ps_control.py: Contains the implementation of the pseudospectral method for trajectory optimization.
- run_traj_search.py: Contains examples of using A* and LT* algorithms for path planning in 2D and 3D environments.
- worlds.py: Contains the implementation of the Gridmap class for creating 2D and 3D environments.
- path_plan.py: Contains an example of using A* and LT* algorithms for path planning in a simple 2D environment.

To use the Python scripts, simply run the desired script with Python.
- run_traj_search.py: script contains examples of using A* and LT* algorithms for path planning in various 2D and 3D environments.
- path_plan.py: script contains an example of using A* and LT* algorithms for path planning in a simple 2D environment.
- pscontrol.py: script contains the implementation of the pseudospectral method for trajectory optimization.

There are several Jupyter Notebooks that were utillized for testing or results vizualization:
- traj_plan.ipynb: experimental notebook with simple implementation of trajectory planning
- psm_check.ipynb: experimental notebook for processing the solution
- worlds.ipynb: experimental notebook with worlds definition and generation
- get_results.ipynb: evaluation of trajectories, generation of pandas tables, and export to LaTeX tables
- print_plots.ipynb: vizualization of trajectories and saving to png

The worlds.py script contains the implementation of the Gridmap class for creating 2D and 3D environments. The generate_some_worlds function in this script can be used to generate various 2D and 3D environments.

## Using the Trajectory Planning Container

To use the Trajectory Planning container, you will need to have Singularity installed on your system. You can download Singularity from the official website: https://docs.sylabs.io/guides/2.6/user-guide/quick_start.html

To run the container, you can use the following command:

```bash
singularity shell traj_container.sif
```

It has build in conda environment `drone_traj`, which after shelling the conteriner is activated using 
```bash
source activate drone_traj.
```
You can then run your Python scripts inside the container.

If you need to access files on your drive from within the container, you can use the --bind option to mount a directory from your system into the container. For example, if you want to mount the /home/user/data directory from your system into the /data directory in the container, you can use the following command:

```bash
singularity shell --bind /home/user/data:/data traj_container.sif
```

This will mount the /home/user/data directory from your system into the /data directory in the container. You can then access the files in the /home/user/data directory from within the container.

## MetaCentrum Computational Grid
Data acrquisition was performed on coputational grid creating bulk jobs using script run_jobs.sh that calls job.sh with specific parameters for environemnt and problem setup.
Config files were generated with Python script gen_traj_params.py
