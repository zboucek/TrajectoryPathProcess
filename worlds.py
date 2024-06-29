#!/usr/bin/env python
# coding: utf-8

# Creator of worlds for the dissertation
#
# The dissertation is dealing with trajectory planning for quadrotor drone. The planning is dealt as Optimal Control Problem that is approached with Pseudospectral multisegment collocation method. The trajectory is built upon the path sought with Lazy Theta Star graph-search method.
#
# There is the list of intended worlds:
# 1. random world with same size spheres (3d)
# 2. forest with randomly distributed trees (3d/2d)
# 3. arboreum with strictly populated lanes of trees (3d/2d)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tikzplotlib as tplt
import random
import os
import sys


class Tree(object):
    """Tree into grid"""

    def __init__(self, file=None, folder=None, height=3):
        """Loading tree sprite to space."""

        # load tree crown from txt
        self.file = file
        self.folder = folder
        if file is None:
            print(f"[WARNING] Path to tree file is unknown")
            return
        elif folder is None:
            with open(os.path.join(file)) as f:
                tree_source = f.read()
        else:
            with open(os.path.join(folder, file)) as f:
                tree_source = f.read()
        self.source = tree_source.splitlines()

        # height of tree from bottom to start of crown
        self.height = height

        line_len = 0
        for line in self.source:
            temp_len = len(line)
            if temp_len > line_len:
                line_len = temp_len

        # transform tree crown to list of leaves and x-z gridmap
        self.list = []
        self.gridmap = np.zeros((len(self.source), line_len))
        self.z_min, self.x_min = 0, 0
        self.z_max, self.x_max = len(self.source)-1, line_len-1
        for i, line in enumerate(self.source):
            for j, char in enumerate(line):
                #         print(f"{i},{j}")
                if char == ' ':
                    self.gridmap[i, j] = 0
                elif char == '@':
                    self.gridmap[i, j] = 1
                    self.list.append([i, j])
        self.list = np.array(self.list)

        # find location for trunk
        find_trunk = []
        for leaf in self.list:
            if leaf[0] == self.z_max:
                find_trunk.append(leaf[1])
        self.trunk = int(np.mean(find_trunk))

        self.total_height = self.z_max + self.height


class Gridmap(object):
    """Create gridmap and map of obstacles"""

    def __init__(self, file=None, folder=None, trees=None,  obstacles=None, start=None, goal=None, n_obstacles=None, limits=None, dimension=None, space=1):
        """Creates gridmap based on txt layout of obstacles and optionally populates map with trees
        """

        # init
        self.file = file    # source file with layout
        self.folder = folder    # folder with source file
        self.start2d = None  # starting position of drone
        self.goal2d = None   # end position of drone
        self.map2d = []  # list of occupied nodes
        self.map3d = None
        self.gridmap3d = None
        self.limits = limits
        self.trees = trees

        self.space = space  # area of the side of the obstacle (for spheres)
        self.n_obstacles = n_obstacles
        self.start = start
        self.goal = goal

        # check dimension
        if dimension is not None:
            self.dim = dimension
        elif obstacles is not None:
            self.dim = obstacles.shape[1]
        elif limits is not None:
            self.dim = int(len(limits)/2)
        elif start is not None:
            self.dim = len(start)
        elif goal is not None:
            self.dim = len(goal)
        elif trees is not None:
            self.dim = 3
        else:
            print(f"[WARNING] dimension is unknown")
            return

        if self.limits is not None:
            self.x_min, self.x_max, self.y_min, self.y_max = self.limits[:4]
            if len(self.limits) > 4:
                self.z_min, self.z_max = self.limits[4:]

        # read file
        if self.file is not None:
            self.read_layout()
        else:
            # OR generate obstacles
            if limits is None:
                self.generate()
            else:
                self.generate(limits=limits)

        # # OR generate obstacles, read file and then merge
        # if self.obstacles is not None and self.file is not None:
        #     self.merge()

        # trees or labyrinth
        if self.trees is not None:
            self.plant_trees()
        elif self.dim == 3 and self.gridmap3d is None:
            self.build_walls()

        self.set_start()
        self.set_goal()

    def generate(self, n_obstacles=random.randint(3, 100), limits=[0, 40, 0, 40, 0, 20]):
        """Generates the world populated randomly with obstacles which are intended in form of spheres.
        """

        # create grid with random obstacles
        if self.n_obstacles is None:
            self.n_obstacles = n_obstacles

        # size of map
        if self.limits is None:
            self.limits = limits
            self.x_min, self.x_max = self.limits[0:2]
            self.y_min, self.y_max = self.limits[2:4]

        # setup grid
        if self.dim == 3:
            if limits is not None:
                self.z_min, self.z_max = self.limits[4:6]
            world_grid = [(x, y, z) for x in range(self.x_min, self.x_max, self.space)
                          for y in range(self.y_min, self.y_max, self.space)
                          for z in range(self.z_min, self.z_max, self.space)]  # grid points
            self.map3d = np.array(
                random.sample(world_grid, self.n_obstacles))

            self.gridmap3d = np.zeros(
                (self.x_max-self.x_min, self.y_max - self.y_min, self.z_max - self.z_min))
            for obstacle in self.map2d:
                self.gridmap3d[obstacle] = 1
        else:
            self.z_min, self.z_max = None, None
            world_grid = [(x, y) for x in range(self.x_min, self.x_max, self.space)
                          for y in range(self.y_min, self.y_max, self.space)]  # grid points
            self.map2d = np.array(
                random.sample(world_grid, self.n_obstacles))

            self.gridmap2d = np.zeros(
                (self.x_max-self.x_min, self.y_max - self.y_min))
            for obstacle in self.map2d:
                self.gridmap2d[obstacle[0],obstacle[1]] = 1

    def read_layout(self):
        """Creates gridmap based on txt layout of obstacles.
        """
        if self.file is None:
            print(f"[WARNING] Path to map file is unknown")
            return
        elif self.folder is None:
            with open(os.path.join(self.file)) as f:
                map_source = f.read()
        else:
            with open(os.path.join(self.folder, self.file)) as f:
                map_source = f.read()
        self.source = map_source.splitlines()

        # map width (y coordinates)
        line_len = 0
        for line in self.source:
            temp_len = len(line)
            if temp_len > line_len:
                line_len = temp_len

        # 2D gridmap with obstacles
        self.gridmap2d = np.zeros((len(self.source), line_len))
        # size of 2D gridmap
        self.x_min, self.y_min = 0, 0
        self.x_max, self.y_max = len(self.source)-1, line_len-1

        # read layout
        for i, line in enumerate(self.source):
            for j, char in enumerate(line):
                # print(f"{i},{j}")
                if char == ' ':
                    self.gridmap2d[i, j] = 0
                elif char == '#':
                    self.gridmap2d[i, j] = 1
                    self.map2d.append([i, j])
                elif char == 'o':
                    self.gridmap2d[i, j] = 2
                    self.start2d = np.array([i, j])
                elif char == 'x':
                    self.gridmap2d[i, j] = 3
                    self.goal2d = np.array([i, j])
        self.map2d = np.array(self.map2d)

    def merge(self, other_map=None):
        """Merge two layouts.
            other_map: Gridmap object will be merged into "self"
        """

        if other_map is None:
            print("Nothing to merge.")
            return

    def plant_trees(self, height=20):
        """Plant trees acording to gridmap"""

        if hasattr(self, "self.z_max"):
            height = self.z_max
        for tree in self.trees:
            if tree.total_height > height:
                height = tree.total_height
        self.z_max = height
        self.z_min = 0

        self.dim = 3

        # if random forest
        # if self.file is None:

        # plant trees in the world and create 3D gridmap
        self.gridmap3d = np.zeros(
            (self.gridmap2d.shape[0], self.gridmap2d.shape[1], height))
        self.map3d = []
        rand_trees = random.choices(self.trees, k=len(self.map2d))
        for i, obstacle in enumerate(self.map2d):
            # fill in map with random trees
            tree = rand_trees[i]
            # bottom position of crown
            x0, y0, z0 = obstacle[0], obstacle[1], tree.height
            for leaf in tree.list:
                # put leaf into map
                if x0+leaf[1]-tree.trunk > self.gridmap3d.shape[0]-1 or x0+leaf[1]-tree.trunk < 0 or y0 > self.gridmap3d.shape[1]-1 or z0+tree.z_max-leaf[0] > self.gridmap3d.shape[2]-1:
                    continue
                if x0 > self.gridmap3d.shape[0]-1 or y0+leaf[1]-tree.trunk > self.gridmap3d.shape[1]-1 or y0+leaf[1]-tree.trunk < 0 or z0+tree.z_max-leaf[0] > self.gridmap3d.shape[2]-1:
                    continue
                self.gridmap3d[x0+leaf[1]-tree.trunk,
                               y0, z0+tree.z_max-leaf[0]] = 1
                self.gridmap3d[x0, y0+leaf[1] -
                               tree.trunk, z0+tree.z_max-leaf[0]] = 1
                self.map3d.append(
                    (x0+leaf[1]-tree.trunk, y0, z0+tree.z_max-leaf[0]))
                self.map3d.append(
                    (x0, y0+leaf[1]-tree.trunk, z0+tree.z_max-leaf[0]))
            for z in range(0, z0):
                # put trunk into map
                if x0 > self.gridmap3d.shape[0]-1 or y0 > self.gridmap3d.shape[1]-1:
                    continue
                self.gridmap3d[x0, y0, z] = 1
                self.map3d.append((x0, y0, z))

        self.map3d = np.array(self.map3d)

    def build_walls(self, height=20):
        """Copy 2D grid layout into 3D grid"""

        self.z_min = 0
        self.z_max = height

        # fake gridmap
        self.gridmap3d = np.zeros(
            (self.gridmap2d.shape[0], self.gridmap2d.shape[0], self.z_max))
        self.gridmap3d = np.repeat(
            self.gridmap2d[:, :, np.newaxis], self.z_max, axis=2)

    def set_start(self, start=None):
        """ Set start location """

        if start is not None:
            self.start = start
        else:
            scale = 0.5
            scale_z = 0.5
            if self.dim == 3:
                start_grid = [(x, y, z) for x in range(self.x_min, int(scale*self.x_max), self.space)
                                for y in range(self.y_min, int(scale*self.y_max), self.space)
                                for z in range(self.z_min, int(scale*self.z_max), self.space)]  # grid points
            else:
                start_grid = [(x, y) for x in range(self.x_min, int(scale*self.x_max), self.space)
                                for y in range(self.y_min, int(scale*self.y_max), self.space)]  # grid points
            # drone start coordinates
            rand_point = np.array(random.sample(start_grid, 1), int)
            while not self.is_free(rand_point[0]):
                rand_point = np.array(random.sample(start_grid, 1), int)

            if rand_point.ndim == 2:
                rand_point = rand_point[0,:]
            
            if hasattr(self, 'start2d'):
                if self.start2d is not None:
                    if self.dim == 3:
                        self.start = np.array(
                            [self.start2d[0], self.start2d[1], rand_point[2]])
                    else:
                        self.start = np.copy(self.start2d)
                else:
                    if self.dim == 2:
                        self.start2d = rand_point
                    else:
                        self.start = np.array(rand_point)
            else:
                if self.dim == 2:
                    self.start2d = rand_point
                else:
                    self.start = np.array(rand_point)

    def set_goal(self, goal=None):
        """ Set goal location """

        if goal is not None:
            self.goal = goal
        else:
            scale = 0.7
            scale_z = 0.5
            if self.dim == 3:
                goal_grid = [(x, y, z) for x in range(int(scale*self.x_max), self.x_max, self.space)
                                for y in range(int(scale*self.y_max), self.y_max, self.space)
                                for z in range(self.z_min, int(scale_z*self.z_max), self.space)]  # grid points
            else:
                goal_grid = [(x, y) for x in range(int(scale*self.x_max), self.x_max, self.space)
                                for y in range(int(scale*self.y_max), self.y_max, self.space)]  # grid points
            # drone goal coordinates
            rand_point = np.array(random.sample(goal_grid, 1), int)
            while not self.is_free(rand_point[0]):
                rand_point = np.array(random.sample(goal_grid, 1), int)


            if rand_point.ndim == 2:
                rand_point = rand_point[0,:]
            
            if hasattr(self, 'goal2d'):
                if self.goal2d is not None:
                    if self.dim == 3:
                        self.goal = np.array(
                            [self.goal2d[0], self.goal2d[1], rand_point[2]])
                    else:
                        self.goal = np.copy(self.goal2d)
                else:
                    if self.dim == 2:
                        self.goal2d = rand_point
                    else:
                        self.goal = np.array(rand_point)
            else:
                if self.dim == 2:
                    self.goal2d = rand_point
                else:
                    self.goal = np.array(rand_point)

    def is_free(self, coordinates):
        """Check if point is free."""
        if coordinates is None:
            return False
        if self.dim == 3:
            return self.gridmap3d[coordinates[0], coordinates[1], coordinates[2]] == 0
        else:
            return self.gridmap2d[coordinates[0], coordinates[1]] == 0

    def draw_cube(self, center, size):
        """ Function for drawing a cube in 3D plot. """
        # Define the vertices that compose the cube
        r = [-size / 2, size / 2]
        vertices = []
        for dx in r:
            for dy in r:
                for dz in r:
                    vertices.append((center[0] + dx, center[1] + dy, center[2] + dz))
                    
        # Define the cube faces using the vertices
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],
            [vertices[4], vertices[5], vertices[7], vertices[6]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[2], vertices[6], vertices[4]],
            [vertices[1], vertices[3], vertices[7], vertices[5]],
        ]
        return Poly3DCollection(faces, facecolors='C0', alpha=0.25)
    
    def plot(self, figsize=(6,4)):
        fig = plt.figure(figsize=figsize)
        scat_size = 20
        if self.dim == 3:
            # plot 3D map
            ax = fig.add_subplot(111,projection="3d")
            # fig.set_figwidth(8)
            # fig.set_figheight(8)
            if self.map3d is not None:
                for center in self.map3d:
                    cube = self.draw_cube(center, self.space)
                    ax.add_collection3d(cube)
                # ax.scatter3D(
                #     self.map3d[:, 0],
                #     self.map3d[:, 1],
                #     self.map3d[:, 2],
                #     marker='o',
                #     s=scat_size, facecolor='lightblue', edgecolor='navy')
                if self.start.ndim == 2:
                    start = self.start[0, :]
                else:
                    start = self.start
                if self.goal.ndim == 2:
                    goal = self.goal[0, :]
                else:
                    goal = self.goal
                ax.scatter3D(
                    start[0],
                    start[1],
                    start[2],
                    marker='x',
                    color='red',
                    s=scat_size
                )
                ax.scatter3D(
                    goal[0],
                    goal[1],
                    goal[2],
                    marker='x',
                    color='green',
                    s=scat_size
                )
                ax.set_box_aspect([1,1,1])  # Equal aspect ratio
                # Set the limits of the axes to fit the range of your data
                xmin, xmax = np.min(self.map3d[:, 0]), np.max(self.map3d[:, 0])
                ymin, ymax = np.min(self.map3d[:, 1]), np.max(self.map3d[:, 1])
                zmin, zmax = np.min(self.map3d[:, 2]), np.max(self.map3d[:, 2])
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_zlim([zmin, zmax])
            else:
                print(f"[WARNING] 3D map not found")
                return
            ax.set_zlabel('$z$ [m]')
            # ax.set_box_aspect((np.ptp(self.map3d[:, 0]), np.ptp(
            #     self.map3d[:, 1]), np.ptp(self.map3d[:, 2])))
        elif self.dim == 2:
            # plot 2D map
            ax = plt.axes()
            # fig.set_figwidth(8)
            # fig.set_figheight(8)
            ax.set_aspect('equal')
            if self.map2d is not None:
                for center in self.map2d:
                    square = patches.Rectangle((center[0]-self.space/2, center[1]-self.space/2), 
                                self.space, self.space,facecolor='C0', edgecolor='C0',alpha=0.5)
                    ax.add_patch(square)
                # old plotting of obstacles
                # ax.scatter(
                #     self.map2d[:, 0],
                #     self.map2d[:, 1],
                #     marker='o',
                #     s=scat_size)
                ax.scatter(
                    self.start2d[0],#-self.space/2,
                    self.start2d[1],#-self.space/2,
                    marker='x',
                    color='red',
                    s=scat_size
                )
                ax.scatter(
                    self.goal2d[0],#-self.space/2,
                    self.goal2d[1],#-self.space/2,
                    marker='x',
                    color='green',
                    s=scat_size
                )
            else:
                print(f"[WARNING] 2D map not found")
                return

            ax.set_aspect('equal', 'box')
        else:
            print(f"[WARNING] dimension is unknown")
            return
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        return ax

    def save(self, folder="", name="gridmap"):
        """Save path, start, goal and computational time."""
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        path_name = os.path.join(folder, name+'.npz')
        if self.dim == 3:
            np.savez(path_name, gridmap=self.gridmap3d,
                     map=self.map3d, start=self.start, end=self.goal, space=self.space)
        else:
            np.savez(path_name, gridmap=self.gridmap2d, map=self.map2d,
                     start=self.start2d, end=self.goal2d, space=self.space)

        # import pickle

        # data = [self.path, self.start_node.position,
        #         self.end_node.position, self.end_time]
        # with open(folder+"/"+name+".pkl", 'wb') as f:
        #     pickle.dump(data, f)

    def save_obj(self, folder="", name="gridmap"):
        """Save object of Gridmap class using Pickle."""
        import pickle
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder,name+".pkl"), 'wb') as f:
            pickle.dump(self, f)

def generate_worlds(print_plots=True, save_results=False):
    folder = 'traj_plan/worlds'
    name = ['walls', 'orchard', 'columns', 'random_spheres', 'forest', 'random_columns']

    trees = [Tree('tree_crown_1.txt', folder, 2), Tree(
        'tree_crown_2.txt', folder, 3), Tree('tree_crown_4.txt', folder, 1)]
    worlds = []
    worlds.append(Gridmap('world_walls.txt', folder,  dimension=2))  # maze
    worlds.append(Gridmap('world_orchard.txt', folder, trees, dimension=3))  # orchard
    worlds.append(Gridmap('world_orchard.txt', folder,  dimension=2))  # columns
    worlds.append(Gridmap(dimension=3, n_obstacles=300))  # random spheres
    worlds.append(Gridmap(dimension=2, trees=trees, limits=[
                    0, 50, 0, 50, 0, 50], n_obstacles=50))  # random forest
    worlds.append(Gridmap(n_obstacles=50,  dimension=2))  # random columns
    
    # save and plot
    for i, world in enumerate(worlds):
        if print_plots:
            world.plot()
        if save_results:
            world.save_obj(folder, name[i])
            world.save(folder, name[i])
    plt.show()
    
def generate_some_worlds(name_files = [], print_plots=True, save_results=False):
    
    current_folder = os.path.dirname(sys.argv[0])
    folder = os.path.join(current_folder,'worlds')
    name = ['simple','simple2','simple3','walls', 'orchard', 'columns', 'random_spheres', 'forest', 'random_columns']

    trees = [Tree('tree_crown_1.txt', folder, 2), Tree(
        'tree_crown_2.txt', folder, 3), Tree('tree_crown_4.txt', folder, 1)]
    worlds = []
    worlds.append(Gridmap('world_simple.txt', folder,  dimension=2))  # simple
    worlds.append(Gridmap('world_simple2.txt', folder,  dimension=2))  # simple2
    worlds.append(Gridmap('world_simple3.txt', folder,  dimension=2))  # simple3
    worlds.append(Gridmap('world_walls.txt', folder,  dimension=2))  # maze
    worlds.append(Gridmap('world_orchard.txt', folder, trees, dimension=3))  # orchard
    worlds.append(Gridmap('world_orchard.txt', folder,  dimension=2))  # columns
    worlds.append(Gridmap(dimension=3, n_obstacles=150, limits=[
                    0, 10, 0, 10, 0, 5]))  # random spheres
    worlds.append(Gridmap(dimension=2, trees=trees, limits=[
                    0, 20, 0, 20, 0, 20], n_obstacles=10))  # random forest
    worlds.append(Gridmap(n_obstacles=30,  dimension=2, limits=[
                    0, 10, 0, 10]))  # random columns
    
    # save and plot
    for i, world in enumerate(worlds):
        if not name[i] in name_files:
            continue
        if print_plots:
            world.plot()
        if save_results:
            world.save_obj(folder, name[i])
            world.save(folder, name[i])
    plt.show()

if __name__ == '__main__':
    # generate_worlds(print_plots=True, save_results=False)
    # generate_some_worlds(name_files = ['simple3','walls', 'orchard', 'columns','random_columns','forest','random_spheres'], print_plots=False, save_results=True)
    generate_some_worlds(name_files = ['forest','random_spheres'], print_plots=False, save_results=True)
    # generate_some_worlds(name_files = ['random_spheres'], print_plots=False, save_results=True)
    # generate_some_worlds(name_files = ['simple2','simple3'], print_plots=False, save_results=True)
    # generate_some_worlds(name_files = ['columns','orchard','random_spheres','random_columns'], print_plots=False, save_results=True)