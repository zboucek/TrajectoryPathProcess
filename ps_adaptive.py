from ps_control import PsControl
from ps_solution import PsSolution
from ps_init import PsInit
from drone_params import Drone
from path_plan import AStar, LTStar
from worlds import Tree, Gridmap, generate_some_worlds, generate_worlds
import numpy as np
from scipy.signal import argrelextrema
import pickle
import cloudpickle
import os
import sys
import copy
import csv
import time
from typing import Optional


class PsAdaptive:
    def __init__(self,
                 drone: Drone = Drone(),
                 world: Optional[Gridmap] = None,
                 ps_init: Optional[PsInit] = None,
                 rel_tol: float = 2.0,
                 abs_tol: float = 1e-3,
                 p_max: float = float("inf"),
                 p_min: int = 10,
                 folder: str = os.path.dirname(sys.argv[0]),
                 file_name: str = "default",
                 max_iter: int = 100,
                 warmstart: bool = True,
                 save: bool = False,
                 n_simpson: int = 10,
                 multisplitting: bool = False,
                 use_obstacles: bool = True,
                 fix_solutions=False,
                 psm_approx: str = 'chebyshev',
                 knot_control_derivative: bool = True,
                 ps_max_iter: int = 10) -> None:
        """
        Initialize PsAdaptive object.

        Args:
            drone: Drone with parameters and constraits of drone and functions for quaternion operations.
            world: Gridmap with start, end and obstacles.
            ps_init: An optional intial guess.
            rel_tol: Relative tolerance for solution.
            abs_tol: Absolute tolerance for solution.
            p_max: Highest degree of polynomials.
            p_min: Lowest degree of polynomials.
            folder: A string representing the folder path.
            file_name: A string representing the file name.
            max_iter: Maximum number of iterations.
            warmstart: A boolean indicating whether to use warmstart for the nonlinear program solver.
            save: A boolean indicating whether to save results.
            n_simpson: An integer representing the number of simpson points in the numerical evaluation of error.
            multisplitting: A boolean indicating whether to split segment according to global maximum or local maxima.
        """
        self.world = world
        self.ps_init = ps_init
        if ps_init is not None:
            self.drone = ps_init.drone
            self.n_col = ps_init.n_col
            self.n_seg = ps_init.n_seg
        else:
            self.drone = drone
            self.n_col = np.array([p_min])
            self.n_seg = len(self.n_col)
            self.ps_init = PsInit(
                np.array([self.drone.x0[:3], self.drone.xf[:3]]), drone, n_col=self.n_col)
            self.ps_init.take_guess()
        self.n_simpson = n_simpson
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.p_max = p_max
        self.p_min = p_min
        self.max_iter = max_iter
        self.ps_max_iter = ps_max_iter
        self.warmstart = warmstart
        self.folder = folder
        self.file_name = file_name
        self.current_iter_no = 0
        self.save = save
        self.start_time = 0
        self.end_time = 0
        self.multisplitting = multisplitting
        self.use_obstacles = use_obstacles
        self.fix_solutions = fix_solutions
        self.psm_approx = psm_approx
        self.knot_control_derivative = knot_control_derivative
        
    def pseudospectral(self, n_col=np.array([20], dtype=int), ps_init=None):
        """
        This function is used to solve a pseudospectral optimization problem.

        Parameters:
        n_col (np.array): An array specifying the number of collocation points for each segment. Default is np.array([20]).
        ps_init (None or object): Initial guess for the pseudospectral problem. Default is None.

        Returns:
        ps_solution (object): The solution of the pseudospectral problem.
        model (object): The solved model of the pseudospectral problem.
        """

        # Initialize flag for time error in the solution
        time_error = True
        max_iter = self.max_iter

        # Loop until a solution with correct time vector is found
        while time_error:
            # Create and set an instance of PsControl
            psc = PsControl()
            model = psc.ps_build(drone=self.drone, obs=self.world,
                                 n_col=n_col, init=ps_init, use_obstacles=self.use_obstacles, psm_approx=self.psm_approx, knot_control_derivative=self.knot_control_derivative)

            # Create an instance of Ps Solution
            ps_solution = PsSolution()
            # Solve the pseudospectral model and update the solution
            psc.ps_solve(model=model, max_iter=max_iter,
                         log_path=os.path.join(self.folder, 'data_traj','logs','ps_log_' +
                         self.file_name+"_"+str(self.current_iter_no).zfill(3)),
                         warmstart=self.warmstart)
            ps_solution.new_solution(model, self.drone, self.world, fit='polyfit', n_simpson=int(
                self.n_simpson), fix_solution=self.fix_solutions)

            # Check if there is a time error in the solution and increase the maximum number of iterations
            time_error = ps_solution.time_error
            max_iter = int(max_iter * 1.2)
        # Save model and solution using cloudpickle
        self.save_results(model, ps_solution)

        return ps_solution, model

    def save_results(self, model, solution):
        """
        Save the model and solution to disk if the `save_results` flag is set.

        Parameters:
            model (object): The Pyomo model to be saved.
            solution (object): The solution to OCP to be saved.

        Returns:
            None
        """
        if self.save_results:
            # Save model and solution using cloudpickle
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
            path_model = self.log_folder+'/ps_' + \
                self.file_name+"_"+str(self.current_iter_no).zfill(3)
            path_solution = self.log_folder+'/ps_sol_' + \
                self.file_name+"_"+str(self.current_iter_no).zfill(3)
            if self.warmstart:
                path_model = path_model+'_with_warmstart'
                path_solution = path_solution+'_with_warmstart'

            with open(path_model+'.pkl', mode='wb') as file:
                cloudpickle.dump(model, file)
            with open(path_solution+'.pkl', mode='wb') as file:
                cloudpickle.dump(solution, file)

    def save_final_results(self, model, solution, iter_time):
        """
        Save the final results of the trajectory planning process.

        Parameters:
            model (object): The optimized model.
            solution (object): The optimized solution.
            iter_time (float): The time taken for the current iteration.

        Returns:
            None
        """
        if self.save_results:
            # Save model and solution using cloudpickle
            save_folder = os.path.join(self.folder,'data_traj')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            path_model = os.path.join(save_folder,'ps_'+self.file_name)
            path_solution = os.path.join(save_folder,'ps_sol_'+self.file_name)
            if self.warmstart:
                path_model = path_model+'_with_warmstart'
                path_solution = path_solution+'_with_warmstart'

            with open(path_model+'.pkl', mode='wb') as file:
                cloudpickle.dump(model, file)
            with open(path_solution+'.pkl', mode='wb') as file:
                cloudpickle.dump(solution, file)

            self.end_time = time.time() - self.start_time
            constraints_error = solution.get_constraints_error()

            # Write log file
            log_file = os.path.join(save_folder,'ps_log_'+self.file_name+'.csv')
            with open(log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow(["Iteration No.", "Objective", "Segmentation", "Absolute Error", "Relative Error",
                                "Total Violation", "State Violation", "Control Violation", "Obstacles Violation", "Total Time", "Iter. Time"])
                writer.writerow([self.current_iter_no, solution.objective, self.n_col, np.max(np.concatenate(solution.max_error)), np.max(np.concatenate(
                    solution.relative_error_max)), constraints_error[0], constraints_error[1], constraints_error[2], constraints_error[3], self.end_time, iter_time])

            # Write log file for collision points
            log_file_col_pts = os.path.join(save_folder,'ps_log_col_pts_'+self.file_name+'.csv')
            with open(log_file_col_pts, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow(["Iteration", "Segmentation",
                                "State", "Control", "Time"])
                writer.writerow([self.current_iter_no, solution.n_col,
                                solution.state, solution.control, solution.time])

    def compute_error(self, solution):
        """
        Compute the absolute and relative errors of the given solution object.

        Args:
            solution: A solution object which has a `get_relative_error` and `get_maximum_error` method.

        Returns:
            A tuple containing the absolute error and the relative error respectively.
        """

        # Calculate the relative error
        rel_err = solution.get_relative_error()

        # Calculate the absolute error
        abs_err = solution.get_absolute_error()

        # Return the errors as a tuple
        return abs_err, rel_err

    def compute_deflection(self, solution):
        """
        Compute the deflection of the solution.

        Args:
            solution: The solution object.

        Returns:
            The deflection value.
        """
        deflection = solution.get_deflection()
        return deflection

    def estimate_degree(self, error_max_seg, eps=1e-5, N_k=None):
        """
        Estimate the required increase in the number of collocation points
        (degree of polynomial) for a given segment based 
        on the maximum error and the desired tolerance.

        Implemented according to the paper 
        Huang, J., Liu, Z., Liu, Z., Wang, Q., & Fu, J. (2019). 
        A pk-Adaptive Mesh Refinement for Pseudospectral Method 
        to Solve Optimal Control Problem. IEEE Access, 7, 161666–161679. 
        https://doi.org/10.1109/ACCESS.2019.2952139

        Opposed to original it is scaled for a higher increase.

        Args:
            error_max_seg (numpy.ndarray): An array of errors for each collocation
                point in the segment.
            eps (float, optional): The desired tolerance. Defaults to 1e-5.
            N_k (int, optional): The number of collocation points for the segment.
                If None, it is calculated as len(error_max_seg). Defaults to None.

        Returns:
            int: The required increase in the number of collocation points for the
            segment to achieve the desired tolerance.
        """

        # Calculate the maximum error for the segment
        e_max = np.max(error_max_seg)

        if N_k is None:
            # Calculate the number of collocation points N(k) for the segment
            N_k = len(error_max_seg)

        # Calculate the required increase in the number of collocation points P(k)
        P_k = np.max([np.ceil(np.emath.logn(N_k, e_max / eps)), 3])
        
        # # The same, but faster
        # P_k = np.max([np.ceil(10*np.emath.logn(N_k, e_max / eps)), 3])

        return P_k

    def run(self):
        self.start_time = time.time()
        guess = copy.deepcopy(self.ps_init)
        self.log_folder = os.path.join(self.folder,'data_traj','iter_data')
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        log_file = os.path.join(self.log_folder,'psa_log_'+self.file_name+'.csv')
        log_file_col_pts = os.path.join(self.log_folder, \
            'psa_log_col_pts_'+self.file_name+'.csv')
        with open(log_file, 'w', newline='', encoding='utf-8') as file1, open(log_file_col_pts, 'w', newline='', encoding='utf-8') as file2:
            writer1 = csv.writer(file1, lineterminator='\n')
            writer2 = csv.writer(file2, lineterminator='\n')
            writer1.writerow(["Iteration", "Objective", "Segmentation",
                             "Absolute Error", "Relative Error", "Total Violation", "State Violation", "Control Violation", "Obstacles Violation", "Loop Time", "PSM Time"])
            writer2.writerow(["Iteration", "Segmentation",
                             "State", "Control", "Time"])
            done = False

            while (not done) and (self.current_iter_no < self.max_iter):
                start_time = time.time()
                self.current_iter_no += 1
                # Solve the pseudospectral problem for the current mesh
                solution, model = self.pseudospectral(self.n_col, guess)
                # Compute the error and deflection for the current solution
                abs_error, rel_error = self.compute_error(solution)
                deflection = self.compute_deflection(solution)

                # Refine the mesh
                if self.multisplitting:
                    # split segment according to several maximum deflection
                    new_mesh, new_time, refine = self.mesh_refinement_relative(
                        solution, deflection, abs_error, rel_error)
                else:
                    # split segment according to one maximum deflection
                    new_mesh, new_time, refine = self.mesh_refinement(
                        solution, deflection, abs_error, rel_error)

                # Check if the new mesh is different from the current mesh
                if new_mesh != self.n_col.tolist() or refine:
                    # Update the current mesh with the new mesh
                    self.n_col = np.array(new_mesh, dtype=int)
                    self.n_seg = int(len(new_mesh))
                    time_sample, state_sample, control_sample = solution.resample_solution(
                        new_time)
                    for i, state in enumerate(self.drone.xf):
                        state_sample[-1][i][-1] = state
                    guess.state_guess = state_sample
                    guess.control_guess = control_sample
                    guess.time_guess = time_sample
                    guess.t_seg_guess = np.zeros(self.n_seg+1)
                    guess.n_col = self.n_col
                    for i in range(self.n_seg):
                        guess.t_seg_guess[i] = new_time[i][0]
                    guess.t_seg_guess[-1] = new_time[-1][-1]
                    print(self.n_col)
                else:
                    # Set the loop condition to True, indicating the refinement is done
                    done = True
                end_time = time.time() - start_time  # Calculate the time taken for the iteration
                
                constraints_error = solution.get_constraints_error()
                # Write iteration data
                writer1.writerow([self.current_iter_no, solution.objective, solution.n_col, np.max(np.concatenate(abs_error)),
                                  np.max(np.concatenate(rel_error)), constraints_error[0], constraints_error[1], constraints_error[2], constraints_error[3], end_time, solution.psm_solve_time])
                writer2.writerow([self.current_iter_no, solution.n_col,
                                 solution.state, solution.control, solution.time])
        self.save_final_results(model, solution, end_time)

        # Return the final solution and mesh
        return solution, model, self.n_col

    def mesh_refinement_relative(self, solution, deflection, abs_error, rel_error):
        new_mesh = []
        new_time = []
        refine = False
        
        if self.psm_approx == 'chebyshev':
            from chebyshev import cheb_scaled
        elif self.psm_approx == 'legendre':
            from legendre import legendre_scaled as cheb_scaled


        # Iterate over each segment in the mesh using zip, enumerate and a for loop
        for i, (current_time, current_n_col) in enumerate(zip(solution.multiseg_time, self.n_col)):
            # Calculate the current polynomial degree
            current_degree = current_n_col - 1

            # Check if the error or deflection exceeds the tolerance
            if np.max(abs_error[i]) > self.abs_tol:
                # Estimate the new polynomial degree based on the error
                p_est = self.estimate_degree(
                    abs_error[i], self.abs_tol, current_n_col)

                # Check if the estimated degree is below the maximum allowable degree
                if np.max(rel_error[i]) > self.rel_tol or current_degree + p_est > self.p_max:
                    # Find the indices of local maxima in the deflection array
                    deflection_arr = np.array(deflection[i])
                    maxima_indices = argrelextrema(
                        deflection_arr, np.greater)[0]
                    split_indices = []
                    threshold = 0.4

                    # if len(maxima_indices) == 1:
                    #      split_indices.append(maxima_indices[0])
                    # else:
                    for index in maxima_indices:
                        if deflection_arr[index] > threshold:
                            split_indices.append(index)
                    if len(split_indices) == 0:
                        split_indices.append(np.argmax(deflection_arr))

                    split_indices.sort()

                    # Check if the first or last index is in the split_indices list
                    if 0 in split_indices:
                        # Remove the first index
                        split_indices = split_indices[1:]
                    if len(current_time) - 1 in split_indices:
                        # Remove the last index
                        split_indices = split_indices[:-1]

                    previous_index = 0
                    for index in split_indices:
                        # Calculate the number of collocation points for the current segment
                        n_col = index - previous_index + 1
                        n_col = max(n_col, self.p_min + 1)

                        # Add the collocation points and time for the current segment
                        new_mesh.append(n_col)
                        grid, _, _ = cheb_scaled(
                            int(n_col - 1), [current_time[previous_index], current_time[index]])
                        new_time.append(grid)

                        refine = True
                        previous_index = index

                    # Add the remaining segment after the last split index
                    n_col_last = current_n_col - previous_index + 1
                    n_col_last = max(n_col_last, self.p_min + 1)
                    new_mesh.append(n_col_last)
                    grid_last, _, _ = cheb_scaled(
                        int(n_col_last - 1), [current_time[previous_index], current_time[-1]])
                    new_time.append(grid_last)

                    if len(split_indices) == 0:
                        if current_degree + p_est <= self.p_max:
                            # Increase the polynomial degree for the current segment
                            new_mesh.append(current_n_col + p_est)
                            grid, __, __ = cheb_scaled(
                                int(current_n_col + p_est - 1), [current_time[0], current_time[-1]])
                            new_time.append(grid)
                        else:
                            new_mesh.append(self.p_max)
                            grid, __, __ = cheb_scaled(
                                int(current_n_col + p_est - 1), [current_time[0], current_time[-1]])
                            new_time.append(grid)

                    refine = True
                else:
                    # Increase the polynomial degree for the current segment
                    new_mesh.append(current_n_col + p_est)
                    grid, __, __ = cheb_scaled(
                        int(current_n_col + p_est - 1), [current_time[0], current_time[-1]])
                    new_time.append(grid)

                    refine = True

            else:
                # Keep the current mesh point and polynomial degree
                new_mesh.append(current_n_col)
                new_time.append(current_time)

        return new_mesh, new_time, refine

    def mesh_refinement(self, solution, deflection, abs_error, rel_error):
        new_mesh = []
        new_time = []
        refine = False
        
        
        if self.psm_approx == 'chebyshev':
            from chebyshev import cheb_scaled
        elif self.psm_approx == 'legendre':
            from legendre import legendre_scaled as cheb_scaled

        # Iterate over each segment in the mesh using zip, enumerate and a for loop
        for i, (current_time, current_n_col) in enumerate(zip(solution.multiseg_time, self.n_col)):
            # Calculate the current polynomial degree
            current_degree = current_n_col - 1

            # Check if the error or deflection exceeds the tolerance
            if np.max(abs_error[i]) > self.abs_tol:
                # Estimate the new polynomial degree based on the error
                p_est = self.estimate_degree(
                    abs_error[i], self.abs_tol, current_n_col)

                # Check if the estimated degree is below the maximum allowable degree
                if np.max(rel_error[i]) > self.rel_tol or current_degree + p_est > self.p_max:
                    # Find the index of local error maxima or local error deflection
                    split_index = np.argmax(deflection[i])
                    # split_index = np.argmax(rel_error[i])
                    split_time = current_time[split_index]

                    if current_time[0] == split_time or split_time == current_time[-1]:
                        # Keep the current mesh point and polynomial degree
                        new_mesh.append(current_n_col)
                        new_time.append(current_time)
                        refine = True
                    else:
                        # Calculate the proportion of collocation points before and after the split index
                        proportion_before = split_index / len(current_time)
                        proportion_after = 1 - proportion_before

                        # Distribute collocation points proportionally, obeying the lowest permitted degree
                        n_col_before = max(
                            int(current_n_col * proportion_before), self.p_min + 1)
                        n_col_after = max(
                            int(current_n_col - n_col_before + 1), self.p_min + 1)

                        # Add the collocation points before and after the split index
                        new_mesh.append(n_col_before)
                        new_mesh.append(n_col_after)
                        grid_before, __, __ = cheb_scaled(
                            int(n_col_before-1), [current_time[0], split_time])
                        new_time.append(grid_before)
                        grid_after, __, __ = cheb_scaled(
                            int(n_col_after-1), [split_time, current_time[-1]])
                        new_time.append(grid_after)

                        refine = True
                else:
                    # Increase the polynomial degree for the current segment
                    new_mesh.append(current_n_col + p_est)
                    grid, __, __ = cheb_scaled(
                        int(current_n_col + p_est - 1), [current_time[0], current_time[-1]])
                    new_time.append(grid)

                    refine = True

            else:
                # Keep the current mesh point and polynomial degree
                new_mesh.append(current_n_col)
                new_time.append(current_time)

        return new_mesh, new_time, refine


class PsAdaptiveSingleSegment(PsAdaptive):
    def __init__(self,
                 drone: Drone = Drone(),
                 world: Optional[Gridmap] = None,
                 ps_init: Optional[PsInit] = None,
                 rel_tol: float = 2.0,
                 abs_tol: float = 1e-3,
                 p_max: float = float("inf"),
                 p_min: int = 10,
                 folder: str = os.path.dirname(sys.argv[0]),
                 file_name: str = "default",
                 max_iter: int = 100,
                 warmstart: bool = True,
                 save: bool = False,
                 n_simpson: int = 10,
                 multisplitting: bool = False,
                 use_obstacles: bool = True,
                 fix_solutions=False,
                 psm_approx: str = 'chebyshev',
                 ps_max_iter: int = 10) -> None:
        """
        Initialize the PsAdaptiveSingleSegment class.

        Args:
            drone: Drone with parameters and constraits of drone and functions for quaternion operations.
            world: Gridmap with start, end and obstacles.
            ps_init: An optional intial guess.
            rel_tol: Relative tolerance for solution.
            abs_tol: Absolute tolerance for solution.
            p_max: Highest degree of polynomials.
            p_min: Lowest degree of polynomials.
            folder: A string representing the folder path.
            file_name: A string representing the file name.
            max_iter: Maximum number of iterations.
            warmstart: A boolean indicating whether to use warmstart for the nonlinear program solver.
            save: A boolean indicating whether to save results.
            n_simpson: An integer representing the number of simpson points in the numerical evaluation of error.
            multisplitting: A boolean indicating whether to split segment according to global maximum or local maxima.
        """
        super().__init__(drone=drone, world=world, ps_init=ps_init, rel_tol=rel_tol, abs_tol=abs_tol, p_max=p_max, p_min=p_min,
                         folder=folder, file_name=file_name, max_iter=max_iter, warmstart=warmstart, save=save, n_simpson=n_simpson, use_obstacles=use_obstacles, fix_solutions=fix_solutions, psm_approx=psm_approx, ps_max_iter=ps_max_iter)
        if self.n_seg > 1:
            raise ValueError("Warning: Too many segments for single segment!")

    def mesh_refinement(self, solution, deflection, abs_error, rel_error):
        """
        Perform mesh refinement based on absolute error and increase the polynomial degree appropriately.

        Args:
            solution: Solution object.
            deflection: Deflection values.
            abs_error: Absolute error values.
            rel_error: Relative error values.

        Returns:
            tuple: New mesh, new time, and refine flag.
        """
        new_mesh = []
        new_time = []
        refine = False
        
        if self.psm_approx == 'chebyshev':
            from chebyshev import cheb_scaled
        elif self.psm_approx == 'legendre':
            from legendre import legendre_scaled as cheb_scaled

        # Iterate over each segment in the mesh using zip, enumerate and a for loop
        for (current_time, current_n_col) in zip(solution.multiseg_time, self.n_col):
            # Calculate the current polynomial degree
            current_degree = current_n_col - 1

            # Check if the error or deflection exceeds the tolerance
            if np.max(abs_error[0]) > self.abs_tol:
                # Estimate the new polynomial degree based on the error
                p_est = self.estimate_degree(
                    abs_error[0], self.abs_tol, current_n_col)

                # Check if the estimated degree is below the maximum allowable degree
                if current_degree + p_est < self.p_max:
                    # Increase the polynomial degree for the current segment
                    new_mesh.append(current_n_col + p_est)
                    grid, __, __ = cheb_scaled(
                        int(current_n_col + p_est - 1), [current_time[0], current_time[-1]])
                    new_time.append(grid)
                    refine = True
                elif current_degree + p_est > self.p_max and current_degree < self.p_max:
                    current_degree = self.p_max

                    # Increase the polynomial degree to max for the current segment
                    new_mesh.append(current_degree)
                    grid, __, __ = cheb_scaled(
                        int(current_degree - 1), [current_time[0], current_time[-1]])
                    new_time.append(grid)
                    refine = True
                else:
                    new_mesh.append(current_n_col)
                    new_time.append(current_time)
            else:
                # Keep the current mesh point and polynomial degree
                new_mesh.append(current_n_col)
                new_time.append(current_time)

        return new_mesh, new_time, refine


def example_in_single_segment(print_plots=True, save_results=True,
                              file_names=['simple', 'orchard', 'columns', 'random_spheres',
                                          'forest', 'random_columns', 'walls'],
                              folder='worlds', n_col=np.array([5e2], dtype=int), max_iter=100, warmstart=True, psm_approx='chebyshev'):

    current_folder = os.path.dirname(sys.argv[0])

    # loading the world
    for file_name in file_names:
        drone = Drone()
        ps_solution = PsSolution()
        # Open the file in binary mode
        with open(os.path.join(current_folder,folder,file_name+'.pkl'), 'rb') as file:
            # Call load method to deserialize
            world = pickle.load(file)

        world.space = 1.0
        # world limits, start and goal
        drone.x_min[0] = world.x_min-world.space-1
        drone.x_max[0] = world.x_max+world.space+1
        drone.x_min[1] = world.y_min-world.space-1
        drone.x_max[1] = world.y_max+world.space+1
        if world.dim == 2:
            drone.x0[0:2] = world.start
            drone.xf[0:2] = world.goal
            drone.x0[2] = 1.0
            drone.x0[2] = 1.0
        else:
            drone.x0[0:3] = world.start
            drone.xf[0:3] = world.goal
            drone.x_min[2] = world.z_min-1
            drone.x_max[2] = world.z_max+1

        # load optimal path to init the problem
        lt_data = np.load(os.path.join(current_folder,'data_path','lt_'+file_name+'.npz'))
        path = lt_data['path']
        start = lt_data['start']
        end = lt_data['end']
        time = lt_data['time']
        # max_iter=50
        # set bounds for polynomial degree
        p_min = 20
        p_max = 500
        # Generate initial guess of problem based on path and drone parameters
        ps_init = PsInit(path_waypoints=path, drone=drone,
                         p_min=n_col[0], constraints=False, boundary=False, psm_approx = psm_approx, single_segment = True)
        ps_init.take_guess()
        # solve the problem
        ps_adaptive = PsAdaptiveSingleSegment(drone=drone, world=world, ps_init=ps_init, abs_tol=1e-2,
                                              p_min=p_min, p_max=p_max, max_iter=max_iter,
                                              warmstart=warmstart, save=save_results, use_obstacles=True, fix_solutions=False, psm_approx = psm_approx)
        ps_solution, model, n_col = ps_adaptive.run()

        if print_plots:
            # plot the solution
            ps_solution.plot_sampled_with_col()


def example(print_plots=True, save_results=True,
            file_names=['simple', 'orchard', 'columns', 'random_spheres',
                        'forest', 'random_columns', 'walls'],
            folder='worlds', n_col=np.array([50], dtype=int), max_iter=100, warmstart=True,
            psm_approx = 'chebyshev'):

    current_folder = os.path.dirname(sys.argv[0])
    
    # loading the world
    for file_name in file_names:
        drone = Drone()
        ps_solution = PsSolution()
        # Open the file in binary mode
        with open(os.path.join(current_folder, folder, file_name+'.pkl'), 'rb') as file:
            # Call load method to deserialize
            world = pickle.load(file)

        world.space = 1.0
        # world limits, start and goal
        drone.x_min[0] = world.x_min-world.space-1
        drone.x_max[0] = world.x_max+world.space+1
        drone.x_min[1] = world.y_min-world.space-1
        drone.x_max[1] = world.y_max+world.space+1
        if world.dim == 2:
            drone.x0[0:2] = world.start
            drone.xf[0:2] = world.goal
            drone.x0[2] = 1.0
            drone.x0[2] = 1.0
        else:
            drone.x0[0:3] = world.start
            drone.xf[0:3] = world.goal
            drone.x_min[2] = world.z_min-1
            drone.x_max[2] = world.z_max+1

        # load optimal path to init the problem
        lt_data = np.load(os.path.join(current_folder,'data_path','lt_'+file_name+'.npz'))
        path = lt_data['path']
        start = lt_data['start']
        end = lt_data['end']
        time = lt_data['time']
        # set bounds for polynomial degree
        p_min = 10
        p_max = 200
        # path=None
        # Generate initial guess of problem based on path and drone parameters
        ps_init = PsInit(path_waypoints=path, drone=drone,
                         p_min=n_col[0], constraints=True, boundary=True, psm_approx = psm_approx)
        ps_init.take_guess(guess_level='position')
        # solve the problem
        ps_adaptive = PsAdaptive(drone=drone, world=world, ps_init=ps_init, abs_tol=1e-2, rel_tol=2.5, p_min=p_min,
                                 p_max=p_max, max_iter=max_iter, warmstart=warmstart, save=save_results,
                                 multisplitting=True, use_obstacles=True, fix_solutions=False, psm_approx=psm_approx)
        ps_solution, model, n_col = ps_adaptive.run()

        if print_plots:
            # plot the solution
            ps_solution.plot_sampled_with_col()


if __name__ == '__main__':
    psm_method = ['legendre', 'chebyshev']
    # multiple segments
    n_col = np.array([10], dtype=int)
    max_iter = 500
    example(print_plots=True, save_results=True, file_names=[
        'simple'], n_col=n_col, max_iter=max_iter, warmstart=True, psm_approx=psm_method[1])

    # single segment
    # n_col = np.array([30], dtype=int)
    # max_iter = 100
    # example_in_single_segment(print_plots=True, save_results=True, file_names=[
    #     'simple'], n_col=n_col, max_iter=max_iter, warmstart=True, psm_approx=psm_method[1])
