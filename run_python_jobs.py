from run_traj_search import find_trajectory_adaptive
from path_plan import AStar, LTStar
from worlds import Tree, Gridmap, generate_some_worlds, generate_worlds
from drone_params import Drone
from ps_control import PsControl
from ps_solution import PsSolution
from ps_init import PsInit
from ps_adaptive import PsAdaptive, PsAdaptiveSingleSegment
import pickle
import cloudpickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import argparse
import json

# Set up argparse to get parameters   
parser = argparse.ArgumentParser()
parser.add_argument("--n_col", type=int, default=15)
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--abs_tol", type=float, default=1e-2)
parser.add_argument("--warmstart", type=bool, default=True)
parser.add_argument("--single_segment", type=bool, default=True)
parser.add_argument("--constraints", type=bool, default=True)
parser.add_argument("--boundary", type=bool, default=True)
parser.add_argument("--init_control", type=bool, default=False)
parser.add_argument("--init_single", type=bool, default=True)
parser.add_argument("--init_level", type=str, choices=['none','time','position', 'speed', 'orientation', 'angular_rate'], default="position") 
parser.add_argument("--world_name", type=str, choices=['simple', 'simple2', 'orchard', 'columns', 'random_spheres',
                        'forest', 'random_columns', 'walls'],default='simple')
parser.add_argument("--method", type=str, choices=["legendre","chebyshev"], default="chebyshev") 
parser.add_argument("--config", type=str, help="JSON config file")

args = parser.parse_args()

# Load config file if provided
if args.config:
    with open(args.config) as f:
        config = json.load(f) 
    # Override with command line args
    vars(args).update(config)

find_trajectory_adaptive(print_plots=False, save_results=True, file_names=[args.world_name], n_col=[args.n_col], abs_tol = args.abs_tol, 
                        max_iter=args.max_iter, warmstart=args.warmstart, psm_approx=args.method,
                        single_segment=args.single_segment, init_enforce_constraints = args.constraints, 
                        init_enforce_boundary=args.boundary, init_level=args.init_level, init_control=args.init_control, init_single=args.init_single)