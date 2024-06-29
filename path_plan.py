# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
# modified to search path in 3D (for drone - alldirectional) with numpy array position
# and Line of Sight check for Lazy Theta* algorithm
# Lines corresponds to pseudocoude in dissertation
from warnings import warn
import heapq
import numpy as np
import time
import os, sys


class AStar(object):
    """A* algorithm for search of optimal path in graph."""

    def __init__(self, maze, start, end, diagonal=False):
        """ init A* algorithm

        input:
            maze: np array with maze
            start: start coordinates
            end: end coordinates
            diagonal: 'True' if diagonal movement is possible
        """
        # 1: function MAIN
        self.path = []
        self.start_time = 0
        self.end_time = np.inf

        # Initialize both open and closed list
        # 2: open = closed = {}
        self.open_list = []
        self.closed_list = []

        # Create start and end node
        # 4: parent(s_start) = s_start
        self.start_node = self.Node(None, start)
        # 3: g(s st art ) = 0
        self.start_node.g = self.start_node.h = self.start_node.f = 0
        self.start_node.parent = self.start_node
        self.end_node = self.Node(None, end)
        self.end_node.g = self.end_node.h = self.end_node.f = 0

        self.end_node.g = np.inf
        self.end_node.f = self.end_node.g + self.end_node.h
        self.start_node.h = self.distance(self.start_node, self.end_node)
        self.start_node.f = self.start_node.g + self.start_node.h
        # Heapify the open_list and Add the start node
        heapq.heapify(self.open_list)
        # 5: open.Insert(s_start, g(s_start) + h(s_start))
        heapq.heappush(self.open_list, self.start_node)

        self.maze = maze

        # what squares do we search
        if diagonal:
            # with diagonal movement
            if self.maze.ndim == 3:
                self.adjacent_squares = np.array(np.meshgrid(
                    [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=int).T.reshape(-1, self.maze.ndim)
            else:
                self.adjacent_squares = np.array(np.meshgrid(
                    [-1, 0, 1], [-1, 0, 1]), dtype=int).T.reshape(-1, self.maze.ndim)

            for i, row in enumerate(self.adjacent_squares):
                if all(row == np.zeros(self.maze.ndim)):
                    delrow = i
            self.adjacent_squares = np.delete(self.adjacent_squares, delrow, 0)
        else:
            # without diagonals
            if self.maze.ndim == 3:
                self.adjacent_squares = np.array(
                    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
            else:
                self.adjacent_squares = np.array(
                    [[1, 0], [-1, 0], [0, 1], [0, -1]])

    class Node(object):
        """ A node class for pathfinding
        """

        def __init__(self, parent=None, position=None):
            self.parent = parent
            self.position = position

            self.g = np.inf
            self.h = 0
            self.f = np.inf

        def __eq__(self, other):
            return self.position == other.position

        def __repr__(self):
            return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

        # defining less than for purposes of heap queue
        def __lt__(self, other):
            return self.f < other.f

        # defining greater than for purposes of heap queue
        def __gt__(self, other):
            return self.f > other.f

    def return_path(self, current_node):
        """ Return path from origin to current_node.
        """
        self.end_time = time.time() - self.start_time
        path = []
        current = current_node
        while True:
            path.append(current.position)
            if current.parent is current:
                # start node is reached
                break
            current = current.parent

        self.path = np.array(path[::-1])
        return self.path  # Return reversed path

    def distance(self, start, end):
        """ Euclidian distance between two points
        """
        if start is None or end is None:
            return np.nan
        else:
            return np.sqrt(np.sum((end.position-start.position)**2))

    def update_node(self, current, child, end_node, open_list):
        """ Update node

        input:
            current: original goal
            child: goal node
            open_list: list with explored nodes
        """
        # 23: function UpdateVertex(s, s')
        # 24: g_old = g(s')
        # 25: ComputeCost(s, s')
        self.compute_cost(current, child)
        child.h = self.distance(child, end_node)
        child.f = child.g + child.h
        # 26: if g(s') < g_old then
        # if child.g < g_old:
        # 27: if s' in open then
        if any((child == x).all() for x in open_list):
            return
        # 30: open.Insert(s', g(s') + h(s'))
        heapq.heappush(open_list, child)

    def compute_cost(self, current, child):
        """ Edit node according to cost.

        input:
            current: original goal
            child: goal node
        """
        # g(s') = g(s) + c(s, s')
        c = self.distance(current, child)
        # Path 1
        # 34: if g(s) + c(s, s') < g(s') then
        if current.g + c < child.g:
            # 35: parent(s') = s
            child.parent = current
            # 35: g(s') = g(s) + c(s, s')
            child.g = current.g + c

    def find_path(self):
        """ Returns a list of tuples as a path from the given start to the given end in the given maze

        output:
            path OR None
        """

        self.start_time = time.time()

        # Adding a stop condition
        outer_iterations = 0
        max_iterations = np.prod(self.maze.shape)*2

        # Loop until you find the end
        # 6: while open!={} do
        while len(self.open_list) > 0:
            outer_iterations += 1

            if outer_iterations > max_iterations:
                # if we hit this point return the path such as it is
                # it will not contain the destination
                warn("giving up on pathfinding too many iterations")
                return self.return_path(current_node)

            # Get the current node
            # 7: s = open.Pop()
            current_node = heapq.heappop(self.open_list)
            # 8: [SetVertex(s)]
            # 12: closed = closed U {s}
            self.closed_list.append(current_node)

            # Found the goal
            # 9: if s = s_goal then
            if not False in (current_node.position == self.end_node.position):
                # 10: return “path found”
                print("found!")
                return self.return_path(current_node)

            # Generate children
            children = []

            # 13: forall s' in nghbr_vis(s) do
            for new_position in self.adjacent_squares:  # Adjacent squares
                # Get node position
                node_position = current_node.position + new_position
                # Make sure within range
                if True in (node_position > self.maze.shape - np.ones(self.maze.ndim)) or True in (node_position < np.zeros(self.maze.ndim)):
                    continue
                # Make sure free space
                if self.maze.ndim == 3:
                    if self.maze[node_position[0], node_position[1], node_position[2]] == 1:
                        continue
                else:
                    if self.maze[node_position[0], node_position[1]] == 1:
                        continue
                # Create new node
                new_node = self.Node(current_node, node_position)
                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if any((child == x).all() for x in self.closed_list):
                    continue

                self.update_node(current_node, child,
                                 self.end_node, self.open_list)

        warn("Couldn't get a path to destination")
        return None

    def save(self, folder="", name="path"):
        """Save path, start, goal and computational time."""

        np.savez(folder+"/"+name+".npz", path=self.path, start=self.start_node.position,
                 end=self.end_node.position, time=self.end_time)

        # import pickle

        # data = [self.path, self.start_node.position,
        #         self.end_node.position, self.end_time]
        # with open(folder+"/"+name+".pkl", 'wb') as f:
        #     pickle.dump(data, f)

    def save_obj(self, folder="", name="path_planner"):
        """Save object of AStar class using Pickle."""
        import pickle

        with open(folder+"/"+name+".pkl", 'wb') as f:
            pickle.dump(self, f)


class LTStar(AStar):
    """Lazy Theta* algorithm for search of optimal path in graph enhanced with line of sight check."""


    def __init__(self, maze, start, end, diagonal=False):
        """ init Lazy Theta* algorithm

        input:
            maze: np array with maze
            start: start coordinates
            end: end coordinates
            diagonal: 'True' if diagonal movement is possible
        """
        super().__init__(maze, start, end, diagonal=diagonal)

    def compute_cost(self, current, child):
        """ Edit node according to cost.

        input:
            current: original goal
            child: goal node
        """
        # g(s') = g(parent(s)) + c(parent(s), s')
        c = self.distance(current.parent, child)
        g_temp = current.parent.g + c
        # Path 2
        # 34: if g(parent(s)) + c(parent(s), s') < g(s') then
        if g_temp < child.g:
            # 35: parent(s') = s
            child.parent = current.parent
            # 35: g(s') = g(s) + c(s, s')
            child.g = g_temp

    def on_sight(self, parent, current):
        """ Check if current node is on sight with its parent.

        uses function from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
        """
        from bresenham3d import Bresenham2D, Bresenham3D
        
        if self.maze.ndim == 3:
            from bresenham3d import Bresenham3D
            line = Bresenham3D(current.position[0], current.position[1], current.position[2],
                                    parent.position[0], parent.position[1], parent.position[2])
            for point in line:
                if self.maze[point[0], point[1], point[2]] == 1:
                    return False
        else:
            from bresenham3d import Bresenham2D
            line = Bresenham2D(current.position[0], current.position[1],
                                    parent.position[0], parent.position[1])

            for point in line:
                if self.maze[point[0], point[1]] == 1:
                    return False

        return True

    def set_node(self, node, children):
        """ Set new node as current and run on sight check.
        """
        # Path 1
        # 2: if not LineOfSight(parent(s),s) then
        if not self.on_sight(node.parent, node):
            # 3: parent = argmin_s' in nghbr_vis(s) ⋂ closed (g(s') + c(s', s))
            eval = np.empty((len(children)))
            eval[:] = np.nan
            for i, child in enumerate(children):
                if any((child == x).all() for x in self.closed_list):
                    # child.g = node.g + \
                    #     distance(child, node)
                    eval[i] = child.g + self.distance(child, node)

            idx = np.nanargmin(eval)

            if eval[idx] != np.nan or eval[idx] != np.inf:
                node.parent = children[idx]
                # 4: g(s) = min_s' in nghbr_vis(s)⋂closed (g(s') + c(s', s))
                node.g = eval[idx]
                node.h = self.distance(node, self.end_node)
                node.f = node.g + node.h

    def nghbr_vis(self, current):
        """Return set of closed nodes visible from current node.
        """
        visible = []
        for node in self.closed_list:
            if node is not current:
                if self.on_sight(current, node):
                    visible.append(node)

        return visible

    def find_path(self):
        """ Returns an array of nodes as a path from the given start to the given end in the given maze

        output:
            path OR None
        """

        self.start_time = time.time()

        # Adding a stop condition
        outer_iterations = 0
        max_iterations = np.prod(self.maze.shape)*2

        # Loop until you find the end
        # 6: while open!={} do
        while len(self.open_list) > 0:
            outer_iterations += 1

            if outer_iterations > max_iterations:
                # if we hit this point return the path such as it is
                # it will not contain the destination
                warn("giving up on pathfinding too many iterations")
                return self.return_path(current_node)

            # Get the current node
            # 7: s = open.Pop()
            current_node = heapq.heappop(self.open_list)
            # 8: [SetVertex(s)]
            # 12: closed = closed U {s}
            self.closed_list.append(current_node)

            visible = self.nghbr_vis(current_node)

            # 8: SetVertex(s)
            self.set_node(current_node, visible)

            # Found the goal
            # 9: if s = s_goal then
            if not False in (current_node.position == self.end_node.position):
                # 10: return “path found”
                print("found!")
                return self.return_path(current_node)

            # Generate children
            children = []

            # 13: forall s' in nghbr_vis(s) do
            for new_position in self.adjacent_squares:  # Adjacent squares
                # Get node position
                node_position = current_node.position + new_position
                # Make sure within range
                if True in (node_position > self.maze.shape - np.ones(self.maze.ndim)) or True in (node_position < np.zeros(self.maze.ndim)):
                    continue
                # Make sure free space
                if self.maze.ndim == 3:
                    if self.maze[node_position[0], node_position[1], node_position[2]] == 1:
                        continue
                else:
                    if self.maze[node_position[0], node_position[1]] == 1:
                        continue
                # Create new node
                new_node = self.Node(current_node, node_position)
                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if any((child == x).all() for x in self.closed_list):
                    continue

                self.update_node(current_node, child,
                                 self.end_node, self.open_list)

        warn("Couldn't get a path to destination")
        return None


def example3d(print_maze=True, save_data=False):

    current_folder = os.path.dirname(sys.argv[0])
    # create map wit huge cubic obstacle in the middle
    maze = np.zeros((10, 10, 10), dtype=int)
    for i in range(3, 8):
        for j in range(3, 8):
            for k in range(3, 8):
                maze[i, j, k] = 1

    start = np.zeros(maze.ndim, dtype=int)
    end = maze.shape - np.ones(maze.ndim, dtype=int)

    astar = AStar(maze, start, end, diagonal=False)
    apath = astar.find_path()
    print(apath)
    print("A* took", astar.end_time, "seconds.")

    ltstar = LTStar(maze, start, end, diagonal=False)
    ltpath = ltstar.find_path()
    print(ltpath)
    print("LT* took", ltstar.end_time, "seconds.")

    if save_data:
        save_folder = os.path.join(current_folder, "data")
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        ltstar.save(save_folder, "ltpath")
        ltstar.save_obj(save_folder, "ltobj")
        astar.save(save_folder, "apath")
        astar.save_obj(save_folder, "aobj")


def example2d(print_maze=True, save_data=False):

    current_folder = os.path.dirname(sys.argv[0])
    # create map wit huge cubic obstacle in the middle
    maze = np.zeros((10, 10), dtype=int)
    for i in range(3, 8):
        for j in range(3, 8):
            maze[i, j] = 1

    start = np.zeros(maze.ndim, dtype=int)
    end = maze.shape - np.ones(maze.ndim, dtype=int)

    astar = AStar(maze, start, end, diagonal=False)
    apath = astar.find_path()
    print(apath)
    if print_maze:
        maze_temp = maze.copy()
        for step in apath:
            maze_temp[step[0]][step[1]] = 2

        for row in maze_temp:
            line = []
            for col in row:
                if col == 1:
                    line.append("\u2588")
                elif col == 0:
                    line.append(" ")
                elif col == 2:
                    line.append(".")
            print("".join(line))
    print("A* took", astar.end_time, "seconds.")

    ltstar = LTStar(maze, start, end, diagonal=False)
    ltpath = ltstar.find_path()
    print(ltpath)
    if print_maze:
        maze_temp = maze.copy()
        for step in ltpath:
            maze_temp[step[0]][step[1]] = 2

        for row in maze_temp:
            line = []
            for col in row:
                if col == 1:
                    line.append("\u2588")
                elif col == 0:
                    line.append(" ")
                elif col == 2:
                    line.append(".")
            print("".join(line))
    print("LT* took", ltstar.end_time, "seconds.")

    if save_data:
        save_folder = os.path.join(current_folder, "data")
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        ltstar.save(save_folder, "ltpath")
        ltstar.save_obj(save_folder, "ltobj")
        astar.save(save_folder, "apath")
        astar.save_obj(save_folder, "aobj")


if __name__ == '__main__':

    example2d(print_maze=False)
    example3d(print_maze=True)
