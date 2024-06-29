import itertools
import json, os, sys


# Generate configs
def generate_configs(method_list =["legendre", "chebyshev"] ):
    
    worlds=['simple', 'simple2', 'orchard', 'columns', 'random_spheres',
                            'forest', 'random_columns', 'walls']
    
    for j, world_name in enumerate(worlds):
        configs = []
        for bool_combo in itertools.product([True, False], repeat=2):
            for method in method_list:
                for init_level in [
                    "none",
                    "time",
                    "position",
                    "speed",
                    "orientation",
                    "angular_rate",
                ]:
                    if init_level == "angular_rate":
                        init_control = [True, False]
                    else:
                        init_control = [False]
                    if bool_combo[0]: 
                        # single segment psm
                        init_single = [True]
                    else:
                        # multiple segment psm
                        init_single = [True, False]
                    for i_ctrl in init_control:
                        for i_single in init_single:
                            config = {
                                "world_name": world_name,
                                "n_col": 15,
                                "max_iter": 1000,
                                "abs_tol": 1e-2,
                                "single_segment": bool_combo[0],
                                "constraints": bool_combo[1],
                                "boundary": bool_combo[1],
                                "warmstart": True,
                                "method": method,
                                "init_level": init_level,
                                "init_control": i_ctrl,
                                "init_single": i_single,
                            }
                            configs.append(config)

        # Save numbered configs
        for i, config in enumerate(configs):
            log_folder = os.path.join(os.path.dirname(sys.argv[0]), "configs")
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            filename = f"config_{j}_{i}.json"
            with open(os.path.join(log_folder, filename), "w") as f:
                json.dump(config, f)

generate_configs(method_list = ['chebyshev'])