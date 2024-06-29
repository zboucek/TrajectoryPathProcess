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


def example_print():
    current_folder = os.path.dirname(sys.argv[0])
    folder = os.path.join(current_folder,'worlds')
    if not os.path.exists(folder):
        os.makedirs(folder)
    # trees = [Tree('tree_crown_1.txt',folder,2)]

    trees = [Tree('tree_crown_1.txt', folder, 2), Tree(
        'tree_crown_2.txt', folder, 3), Tree('tree_crown_4.txt', folder, 1)]
    # trees = [Tree('tree_crown_1.txt',folder,2),Tree('tree_crown_2.txt',folder,3),Tree('tree_crown_3.txt',folder,2),Tree('tree_crown_4.txt',folder,1)]
    grid = Gridmap('world_orchard.txt', folder, trees, dimension=3)
    grid2 = Gridmap('world_orchard.txt', folder, trees=None, dimension=2)
    grid4 = Gridmap('world_walls.txt', folder, trees=None, dimension=2)
    grid3 = Gridmap(dimension=3)
    grid.plot()
    grid2.plot()
    grid4.plot()
    grid3.plot()
    plt.show()


def example_ltstar(atrue=True, lttrue=True, print_plot=True):
    
    current_folder = os.path.dirname(sys.argv[0])
    folder = os.path.join(current_folder,'worlds')
    if not os.path.exists(folder):
        os.makedirs(folder)
    trees = [Tree('tree_crown_1.txt', folder, 2), Tree(
        'tree_crown_2.txt', folder, 3), Tree('tree_crown_4.txt', folder, 1)]
    # grid = Gridmap('world_walls.txt', folder, trees=None, dimension=2)
    grid = Gridmap('world_orchard.txt', folder, trees=None, dimension=3)
    # grid = Gridmap('world_walls.txt', folder, trees=None, dimension=3)

    if atrue:
        astar = AStar(grid.gridmap3d, grid.start, grid.goal, diagonal=False)
        apath = astar.find_path()
        print(apath)
        if print_plot:
            grid.plot()
            path = apath
            print("A* took", astar.end_time, "seconds.")
            if path.shape[1] == 2:
                plt.plot(path[:, 0], path[:, 1], 'r')
            else:
                plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
            plt.show()
    if lttrue:
        ltstar = LTStar(grid.gridmap3d, grid.start, grid.goal, diagonal=False)
        ltpath = ltstar.find_path()
        print(ltpath)
        print("LT* took", ltstar.end_time, "seconds.")
        if print_plot:
            grid.plot()
            path = ltpath
            if path.shape[1] == 2:
                plt.plot(path[:, 0], path[:, 1], 'r')
            else:
                plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
            plt.show()


def example_ltstar_2d(atrue=True, lttrue=True, print_plot=True):
    
    current_folder = os.path.dirname(sys.argv[0])
    folder = os.path.join(current_folder,'worlds')
    # grid = Gridmap('world_walls.txt', folder, trees=None, dimension=2)
    grid = Gridmap('world_orchard.txt', folder, trees=None, dimension=2)

    if atrue:
        astar = AStar(grid.gridmap2d, grid.start2d,
                      grid.goal2d, diagonal=False)
        apath = astar.find_path()
        print(apath)
        if print_plot:
            grid.plot()
            print("A* path:")
            path = apath
            print("A* took", astar.end_time, "seconds.")
            if path.shape[1] == 2:
                plt.plot(path[:, 0], path[:, 1], 'r')
            else:
                plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
            plt.show()
    if lttrue:
        ltstar = LTStar(grid.gridmap2d, grid.start2d,
                        grid.goal2d, diagonal=False)
        ltpath = ltstar.find_path()
        print("LT* path:")
        print(ltpath)
        print("LT* took", ltstar.end_time, "seconds.")
        if print_plot:
            grid.plot()
            path = ltpath
            if path.shape[1] == 2:
                plt.plot(path[:, 0], path[:, 1], 'r')
            else:
                plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
            plt.show()


def find_paths(file_names = ['simple','simple2','simple3','orchard', 'columns', 
                             'random_spheres', 'forest', 'random_columns'], 
               folder_worlds = os.path.join(os.path.dirname(sys.argv[0]),'worlds'), 
               folder_save = os.path.join(os.path.dirname(sys.argv[0]),'data_path')):
    
    for name in file_names:
        print(name)
        npzfile = np.load(folder_worlds+'/'+name+'.npz')
        astar = AStar(npzfile['gridmap'], npzfile['start'], npzfile['end'])
        aresult = astar.find_path()
        ltstar = LTStar(npzfile['gridmap'], npzfile['start'], npzfile['end'])
        ltresult = ltstar.find_path()
        print(ltresult)

    if aresult is None or ltresult is None:
        return None, None
    else:
        astar.save_obj(folder_save, 'a_'+name)
        astar.save(folder_save, 'a_'+name)
        ltstar.save_obj(folder_save, 'lt_'+name)
        ltstar.save(folder_save, 'lt_'+name)
        return aresult.shape[0], ltresult.shape[0]


def find_trajectory(print_plots=True, save_results=True,
                    file_names=['simple','orchard', 'columns', 'random_spheres',
                                'forest', 'random_columns', 'walls'],
                    folder='worlds', n_col=np.array([5e2], dtype=int), max_iter=1000, warmstart = False):
    import pickle
    import cloudpickle
    
    current_folder = os.path.dirname(sys.argv[0])

    drone = Drone()
    ps_solution = PsSolution()
    # generating the world
    for file_name in file_names:
        # Open the file in binary mode
        with open(os.path.join(current_folder,folder,file_name+'.pkl'), 'rb') as file:
            # Call load method to deserialze
            world = pickle.load(file)
            
        world.space = 1.0
        # world limits, start and goal
        drone.x_min[0] = world.x_min-world.space
        drone.x_max[0] = world.x_max+world.space
        drone.x_min[1] = world.y_min-world.space
        drone.x_max[1] = world.y_max+world.space
        if world.dim == 2:
            drone.x0[0:2] = world.start
            drone.xf[0:2] = world.goal
            drone.x0[2] = 1.0
            drone.x0[2] = 1.0
        else:
            drone.x0[0:3] = world.start
            drone.xf[0:3] = world.goal
            drone.x_min[2] = world.z_min
            drone.x_max[2] = world.z_max

        psc = PsControl()
        model = psc.ps_build(drone, world, n_col)
        psc.ps_solve(model = model, max_iter = max_iter, tol=1e-2, log_path=os.path.join(current_folder,'data_traj','ps_log_'+file_name), warmstart= warmstart)
        ps_solution.new_solution(model, drone, world, fit='poly', fix_solution=False)

        if save_results:
            path_model = os.path.join(current_folder,'data_traj','ps_'+file_name)
            path_solution = os.path.join(current_folder,'data_traj','ps_sol_'+file_name)
            if warmstart:
                path_model = path_model+'_with_warmstart'
                path_solution = path_solution+'_with_warmstart'
                
            with open(path_model+'.pkl', mode='wb') as file:
                cloudpickle.dump(model, file)
            with open(path_solution+'.pkl', mode='wb') as file:
                cloudpickle.dump(ps_solution, file)

        if print_plots:
            ps_solution.plot_sampled_with_col()
            
def calc_ratio2arena(world_min, world_max, arena_min, arena_max):
    """
    Scales the given world coordinates to arena coordinates.
    
    Parameters:
        world_min (numpy.ndarray): The minimum world coordinates.
        world_max (numpy.ndarray): The maximum world coordinates.
        arena_min (numpy.ndarray): The minimum arena coordinates.
        arena_max (numpy.ndarray): The maximum arena coordinates.
    
    Returns:
        tuple: A tuple containing the scaled minimum world coordinates, scaled maximum world coordinates,
               the ratio.
    """
    
    arena_center = (arena_min+arena_max)/2
    arena_size = arena_max - arena_min
    world_size = world_max - world_min
    ratio_array = arena_size[:len(world_size)]/world_size
    ratio_idx = np.argmin(ratio_array)
    ratio = ratio_array[ratio_idx]
    
    world_min_scaled = np.zeros(arena_size.shape)
    world_max_scaled = np.zeros(arena_size.shape)
    world_min_scaled[:2] = -ratio*(world_size[:2])/2 + arena_center[:2]
    world_max_scaled[:2] = ratio*(world_size[:2])/2 + arena_center[:2]
    
    if len(world_min) == 2:
        world_min_scaled[2] = arena_min[2]
        world_max_scaled[2] = arena_max[2]
    else:
        world_min_scaled[2] = 0.0
        world_max_scaled[2] = ratio*(world_size[2])
        
    return world_min_scaled, world_max_scaled, ratio
    
def scale_obstacle_position(world, path, scaled_world):
    """
    Scale the obstacle positions in the world and the path based on the scaled_world.

    Args:
        world (World): The original world object containing the obstacle positions.
        path (ndarray): The original path.
        scaled_world (World): The scaled world object.

    Returns:
        Tuple[World, ndarray]: A tuple containing the scaled world object and the scaled path.
    """
    # get ratio (all axes were scaled in same proportion)
    # ratio = (scaled_world.x_max - scaled_world.x_min)/(world.x_max - world.x_min)
    if world.dim == 3:
        center = np.array([scaled_world.x_min, scaled_world.y_min, scaled_world.z_min], float)
        original_map = world.map3d
    else:
        center = np.array([scaled_world.x_min, scaled_world.y_min], float)
        original_map = world.map2d
    # scale path
    scaled_path = path*scaled_world.ratio + center
    # scale obstacle and move it to the center of gridpoint
    scaled_map = original_map*scaled_world.ratio + center
        
    if world.dim == 3:
        scaled_world.map3d = scaled_map
    else:
        scaled_world.map2d = scaled_map
    
    return scaled_world, scaled_path
    
    
def test_scale2lab():
    world_min = np.array([0.0, 0.0])
    world_max = np.array([20.0, 10.0])
    arena_min = np.array([-5.0, -5.0, 0.0])
    arena_max = np.array([5.0, 5.0, 3.0])
    
    print(calc_ratio2arena(world_min, world_max, arena_min, arena_max))
            
def scale_world2lab(drone = Drone(), world = Gridmap(), path = None):
    """Scale the world to the lab according to boudaries given in Drone object

    Args:
        drone (Drone, optional): drone object with boundaries of flight space. Defaults to Drone().
        world (Gridmap, optional): Gridmap, list of obstacles and start, goal points. Defaults to Gridmap().
        path (np.array, optional): path through gridmap. Defaults to None.

    Returns:
        Gridmap, np.array: scaled world and path according to laboratory flight space given in Drone object
    """
    
    if path is None:
        scaled_path = None
    else:
        scaled_path = np.copy(path)
    scaled_world = copy.deepcopy(world)
    if world.dim == 3:
        world_min = np.array([world.x_min, world.y_min, world.z_min])
        world_max = np.array([world.x_max, world.y_max, world.z_max])
    else:
        world_min = np.array([world.x_min, world.y_min])
        world_max = np.array([world.x_max, world.y_max])
    world_min, world_max, ratio = calc_ratio2arena(world_min, world_max, drone.x_min[:3], drone.x_max[:3])
    
    scaled_world.ratio = ratio
    scaled_world.x_min, scaled_world.y_min = world_min[:2]
    scaled_world.x_max, scaled_world.y_max = world_max[:2]
    if world.dim == 3:
        scaled_world.z_min = world_min[2]
        scaled_world.z_max = world_max[2]
    
    if scaled_world.start is None or scaled_world.goal is None:
        scaled_world.start = scaled_world.start2d
        scaled_world.goal = scaled_world.goal2d
    
    scaled_world.start = np.array((scaled_world.start*scaled_world.ratio + world_min[:len(scaled_world.start)]), float) 
    scaled_world.goal = np.array((scaled_world.goal*scaled_world.ratio + world_min[:len(scaled_world.goal)]), float)
    
    if scaled_world.dim == 2:
        scaled_world.start2d = scaled_world.start[:2]
        scaled_world.goal2d = scaled_world.goal[:2]
    
    scaled_world, scaled_path = scale_obstacle_position(world, path, scaled_world)
    scaled_world.space = world.space*scaled_world.ratio
    
    return scaled_world, scaled_path

def test_scale_world2lab(file_name = 'columns'):
    drone = Drone()
    current_folder = os.path.dirname(sys.argv[0])
    folder='worlds'
    with open(os.path.join(current_folder,folder,file_name+'.pkl'), 'rb') as file:
        # Call load method to deserialize
        world = pickle.load(file)
    
    lt_data = np.load(os.path.join(current_folder,'data_path','lt_'+file_name+'.npz'))
    path = lt_data['path']
    print(path, world.start, world.goal)
    world.plot()
    if path.shape[1] == 2:
        plt.plot(path[:, 0], path[:, 1], 'r')
    else:
        plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
        
    scaled_world, scaled_path = scale_world2lab(drone, world = world, path = path)
    print(scaled_path, scaled_world.start, scaled_world.goal)
    scaled_world.plot()
    if scaled_path.shape[1] == 2:
        plt.plot(scaled_path[:, 0], scaled_path[:, 1], 'r')
    else:
        plt.plot(scaled_path[:, 0], scaled_path[:, 1], scaled_path[:, 2], 'r')
    plt.show()
    
def load_world_path_and_scale(world_name = 'simple',folder='worlds', current_folder = os.path.dirname(sys.argv[0]), drone = Drone()):
    
    with open(os.path.join(current_folder,folder,world_name+'.pkl'), 'rb') as file:
            # Call load method to deserialize
            world = pickle.load(file)
            
    # load optimal path to init the problem
    lt_data = np.load(os.path.join(current_folder,'data_path','lt_'+world_name+'.npz'))
    path = lt_data['path']
    
    # return world, path

    scaled_world, scaled_path = scale_world2lab(drone, world = world, path = path)
    if scaled_world.dim == 2:
        drone.x0[0:2] = scaled_world.start
        drone.xf[0:2] = scaled_world.goal
        drone.x0[2] = 1.0
        drone.xf[2] = 1.0
    else:
        drone.x0[0:3] = scaled_world.start
        drone.xf[0:3] = scaled_world.goal
        drone.x_min[2] = scaled_world.z_min
        drone.x_max[2] = scaled_world.z_max
        
    return scaled_world, scaled_path
    
def find_trajectory_adaptive(print_plots=True, save_results=True,
            file_names=['simple', 'simple2', 'orchard', 'columns', 'random_spheres',
                        'forest', 'random_columns', 'walls'],
            folder='worlds', n_col=np.array([30], dtype=int), single_segment = False, max_iter=1e3, warmstart=True,
            psm_approx = 'chebyshev', abs_tol = 1e-2, rel_tol = 2.5, init_enforce_constraints = True, 
            init_enforce_boundary = True, init_level = 'position', init_control = True, init_single = True):

    current_folder = os.path.dirname(sys.argv[0])
    
    # loading the world
    for file_name in file_names:
        drone = Drone()
        ps_solution = PsSolution()
        # Open the file in binary mode
        with open(os.path.join(current_folder,folder,file_name+'.pkl'), 'rb') as file:
            # Call load method to deserialize
            world = pickle.load(file)
        
        if init_enforce_boundary:
            file_bound = 'boundon'
        else:
            file_bound = 'boundoff'
        if init_enforce_constraints:
            file_const = 'conston'
        else:
            file_const = 'constoff'
        if single_segment:
            file_seg = 'single'
        else:
            file_seg = 'multi'
        if init_control:
            file_ctrl = 'ctrlon'
        else:
            file_ctrl = 'ctrloff'
        if init_single:
            file_init_domain = 'isingle'
        else:
            file_init_domain = 'imulty'
            
        file_name_save = file_name+'_'+file_bound+'_'+file_const+'_'+init_level+'_'+file_ctrl+'_'+file_init_domain+'_'+file_seg+'_'+psm_approx
        
        world.space = 1.0

        # load optimal path to init the problem
        lt_data = np.load(os.path.join(current_folder,'data_path','lt_'+file_name+'.npz'))
        path = lt_data['path']
        # start = lt_data['start']
        # end = lt_data['end']
        # time = lt_data['time']
        scaled_world, scaled_path = scale_world2lab(drone, world = world, path = path)
        if scaled_world.dim == 2:
            drone.x0[0:2] = scaled_world.start
            drone.xf[0:2] = scaled_world.goal
            drone.x0[2] = 1.0
            drone.xf[2] = 1.0
        else:
            drone.x0[0:3] = scaled_world.start
            drone.xf[0:3] = scaled_world.goal
            drone.x_min[2] = scaled_world.z_min
            drone.x_max[2] = scaled_world.z_max
        
        # set bounds for polynomial degree
        p_min = n_col[0]-1
        p_max = 500
        # Generate initial guess of problem based on path and drone parameters
        ps_init = PsInit(path_waypoints=scaled_path, drone=drone, interpolation_method = 'spline', 
                        p_min=n_col[0], constraints=init_enforce_constraints, boundary=init_enforce_boundary, 
                        psm_approx = psm_approx, single_segment = init_single)
        ps_init.take_guess(guess_level=init_level, guess_control=init_control)
        ps_init.save(folder=current_folder, file_name=file_name_save)
        # ps_init.take_guess()
        # solve the problem
        if single_segment:
            ps_adaptive = PsAdaptiveSingleSegment(drone=drone, world=scaled_world, ps_init=ps_init, abs_tol=abs_tol,
                                                  p_min=p_min, p_max=p_max, max_iter=max_iter, file_name=file_name_save,
                                                  warmstart=warmstart, save=save_results, use_obstacles=True, fix_solutions=False, psm_approx = psm_approx)
        else:
            ps_adaptive = PsAdaptive(drone=drone, world=scaled_world, ps_init=ps_init, abs_tol=abs_tol, rel_tol=rel_tol, p_min=p_min,
                                    p_max=p_max, max_iter=max_iter, warmstart=warmstart, save=save_results, file_name=file_name_save,
                                    multisplitting=True, use_obstacles=True, fix_solutions=False, psm_approx=psm_approx)
        ps_solution, model, n_col = ps_adaptive.run()

        if print_plots:
            # plot the solution
            ps_solution.plot_sampled_with_col()


def find_random_columns():
    found = False
    while not found:
        generate_some_worlds(
            name_files=['forest'], print_plots=True, save_results=True)
        found = find_paths()
        print(found)
        
def example_adaptive_trajectory_search():
    
    n_col = np.array([15], dtype=int)
    max_iter = 1000
    worlds=['simple', 'simple2', 'orchard', 'columns', 'random_spheres',
                        'forest', 'random_columns', 'walls']
    psm_method = ['legendre', 'chebyshev']
    single_segment = False
    init_enforce_constraints = True
    init_enforce_boundary = True
    warmstart = True
    init_level = ['none','position', 'speed', 'orientation', 'angular_rate']
    init_control = False
    # find_trajectory(print_plots=True, save_results=False, file_names=[
                    # 'simple'], n_col=n_col, max_iter=max_iter, warmstart = True)
    find_trajectory_adaptive(print_plots=True, save_results=True, file_names=[worlds[0]], n_col=n_col, abs_tol = 1e-1, 
                             max_iter=max_iter, warmstart=warmstart, psm_approx=psm_method[1],
                             single_segment=single_segment, init_enforce_constraints = init_enforce_constraints, 
                             init_enforce_boundary=init_enforce_boundary, init_level=init_level[0], init_control=init_control)
    
def example_load_world_path_and_plot(world_name = 'simple3'):
    drone = Drone()
    world, path = load_world_path_and_scale(world_name, current_folder = '', drone = drone)
    print(path)
    world.plot()
    if path.shape[1] == 2:
        plt.plot(path[:, 0], path[:, 1], 'r')
    else:
        plt.plot(path[:, 0], path[:, 1], path[:, 2], 'r')
    plt.show()

if __name__ == '__main__':
    # example_ltstar()
    # example_ltstar_2d()
    # find_random_columns()
    # find_paths()
    # find_paths(['simple3','columns',  'random_columns', 'walls','orchard', 'random_spheres', 'forest'])
    # len_ltpath = 2
    # while len_ltpath < 3:
    world_name = 'simple'
    # generate_some_worlds(name_files = [world_name], print_plots=False, save_results=True)
    # __, len_ltpath = find_paths([world_name])
    # test_scale_world2lab(world_name)
    # example_adaptive_trajectory_search()
    # example_load_world_path_and_plot('walls')
    find_trajectory_adaptive(print_plots=True, save_results=False, file_names=[world_name], n_col=[15], 
                             abs_tol = 1e-2,single_segment= True, init_level = 'position', init_control=False)