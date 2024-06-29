import numpy as np
from scipy.interpolate import CubicSpline
from drone_params import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import lagrange


class PsInit:
    """
    Class for initial solution with state and control vector, both time dependent, 
    variables for path waypoints and array with number of collocation grid for each segment.
    """

    def __init__(self, path_waypoints, drone=Drone(), n_col=None, **kwargs):
        """
        Initialize the PsInit class.

        Parameters
        ----------
        path_waypoints: numpy.ndarray
            Array with shape (num_waypoints, 3) representing the 3D waypoints of the path.
        n_col: list
            List of integers representing the number collocation grid for each segment in the path.
        drone: Drone
            class for drone parameters and boundary conditions
        kwargs: Additional keyword arguments to be passed.
            For 'interpolation_method': str, optional
                Interpolation method to use for fitting the waypoints.
                Supported options: 'polynomial', 'spline'.
                Default: 'spline'
            For 'constraints': bool, optional
                Enforce constraints on the initial guess trajectory.
            For 'boundary': bool, optional
                Enforce boundary conditions on the initial guess trajectory.
            For 'single_segment': bool, optional
                If True, initial guess trajectory will be provied in form of a single segment.
            For 'p_min', provide 'order' as the order of the Chebyshev polynomial.
            For 'spline', provide 's' as the smoothness parameter.
            For 'psm_approx': str, optional
                Type od nodes for initial guess. Choice between Chebyshev Gauss-Lobatto or Legendre Gauss-Lobatto points.
                Supported options: 'chebyshev', 'legendre'.
                Default: 'chebyshev'
        """
        self.kwargs = kwargs
        self.drone = drone
        if path_waypoints is None:
            self.path_waypoints = np.array([[drone.x0[:3]], [drone.xf[:3]]]).T
        else:
            self.path_waypoints = path_waypoints
        self.path_dim = self.path_waypoints.shape[1]
        self.n_col = n_col
        self.interpolation_method = self.kwargs.get(
            'interpolation_method', 'spline')
        self.constraints = self.kwargs.get('constraints', False)
        self.boundary = self.kwargs.get('boundary', False)
        self.single_segment = self.kwargs.get('single_segment', False)
        self.p_min = self.kwargs.get('p_min', 10)
        if self.n_col is None:
            self.n_seg = int(len(self.path_waypoints)-1)
            self.n_col = self.p_min*np.ones(self.n_seg, dtype=int)
        else:
            self.n_seg = len(n_col)
        self.nx = self.drone.nx  # position, velocity, orientation, angular velocity
        self.nu = self.drone.nu  # thrust and torques
        self.n_col_sum = sum(self.n_col)
        self.time_guess = None
        self.state_guess = None
        self.control_guess = None
        self.psm_approx = self.kwargs.get('psm_approx', 'chebyshev')
        # choose collocation time grid accordingly to pseudospectral method
        if self.psm_approx == 'chebyshev':
            from chebyshev import cheb_scaled
        elif self.psm_approx == 'legendre':
            from legendre import legendre_scaled as cheb_scaled
        self.col_points = cheb_scaled
        self.guess_level = None
        self.guess_control = None

    def drone_position(self, t, x0, xf, t0, tf_max):
        """Function to compute the drone's position at a given time"""
        position = x0 + (t - t0) * (xf - x0) / (tf_max - t0)
        return position

    def compute_time_grid(self, path_waypoints=None, velocity_limit=None, t_seg=None):
        """Compute the time grid for the initial guess trajectory

        Args:
            path_waypoints (np.array, optional): np.array with waypoints in rows. Defaults to None.
            velocity_limit (array, optional): velocity limit in 2D or 3D. Defaults to None.
            t_seg (array, optional): time[s] for each knot point between segments + boundary. Defaults to None.
        """
        
        # if the time in segments are not given, choose them
        if t_seg is None:
            if self.guess_level == 'none':
                # time guess is not based on velocity constraints
                t_seg = [self.drone.t0, (self.drone.tf_max + self.drone.tf_min)/2]
                t_seg = np.linspace(t_seg[0], t_seg[-1], self.n_seg+1)
            else:
                # time guess is based on velocity constraints
                if path_waypoints is None:
                    path_waypoints = self.path_waypoints
                if velocity_limit is None:
                    velocity_limit = self.drone.x_max[3:3+path_waypoints.shape[1]]
                # weight path by maximum velocity of drone to get optimistic guess about time of arival
                dt = np.zeros(path_waypoints.shape[0])
                for i, dist in enumerate(np.diff(path_waypoints, axis=0)):
                    dt[i+1] = np.max(np.abs(dist/velocity_limit))
                # calculate optimistic time of arrival
                t_seg = self.drone.t0+np.cumsum(dt)

        # get the time evaluation at the collocation grid
        time_grid = []
        if len(t_seg) == self.n_seg+1:
            # segmented for every neighborhood of waypoints
            for i_seg, n in enumerate(self.n_col):
                grid, __, __ = self.col_points(int(n), t_seg[i_seg:i_seg+2])
                time_grid.append(grid)
        else:
            # linearly divided multisegment
            dt = (self.drone.tf - self.drone.t0) / self.n_seg

            # Generate the vector of time points
            t_seg_new = np.linspace(t_seg[0], t_seg[-1], self.n_seg+1)
            for i_seg, n in enumerate(self.n_col):
                grid, __, __ = self.col_points(int(n), t_seg_new[i_seg:i_seg+2])
                time_grid.append(grid)

        self.time_guess = time_grid
        self.t_seg_guess = t_seg

    def fit_data(self, data, time = None):
        if self.interpolation_method == 'polynomial':
            return self._fit_data_polynomial(data)
        elif self.interpolation_method == 'spline':
            return self._fit_data_spline(data)
        elif self.interpolation_method == 'lagrange':
            return self._fit_data_polynomial_lagrange(data)
        else:
            raise ValueError("Unsupported interpolation method.")

    def _fit_data_polynomial(self, data):
        # Fit the data using polynomials
        if len(self.t_seg_guess) < 2:
            raise ValueError("Number of waypoints must be at least 2.")
        order = len(self.t_seg_guess)-1

        poly = []
        for d in data:
            poly.append(np.polynomial.Polynomial.fit(
                self.t_seg_guess, d, order))

        if len(data) == 1:
            return poly[0]
        else:
            return poly
        
    def _fit_data_polynomial_lagrange(self, data):
        # Fit the data using polynomials
        if len(self.t_seg_guess) < 2:
            raise ValueError("Number of waypoints must be at least 2.")

        poly = []
        for d in data:
            poly.append(lagrange(self.t_seg_guess,d))

        if len(data) == 1:
            return poly[0]
        else:
            return poly

    def _fit_data_spline(self, data):
        # Fit the data using CubicSpline with respect to self.time_guess
        spline = []
        for d in data:
            spline.append(CubicSpline(self.t_seg_guess, d))

        if len(data) == 1:
            return spline[0]
        else:
            return spline

    def _eval_curves_np(self, time, curves):
        return np.array(self._eval_curves(time, curves)).T

    def _eval_curves(self, time, curves):
        curve_eval = []
        for curve in curves:
            curve_eval.append(curve(time))
        return curve_eval

    def _get_derivatives(self, curves, order=1):
        derivatives = []
        for curve in curves:
            derivatives.append(self._get_derivative(curve, order))
        return derivatives

    def _get_derivative(self, curve, order=1):
        if self.interpolation_method in ['polynomial', 'lagrange']:
            return curve.deriv(order)
        elif self.interpolation_method == 'spline':
            return curve.derivative(order)
        else:
            raise ValueError("Unsupported interpolation method.")

    def compute_position_traj(self, path):
        """ Get trajectory of position according to path and time guess.
             Handles exception if the dimmensionality does not fit with the path dimmension   
        """
        try:
            return self.fit_data(path.T)
        except:
            Warning("The path dimensionality does not fit.")
            return self.fit_data(path)

    def _enforce_constraints(self, curve, val_min=None, val_max=None, val_0=None, val_f=None, time=None, control=False):
        if self.boundary or self.constraints:
            if time is None:
                time = self.t_seg_guess

            curve_eval = self._eval_curves_np(time, curve)
            if self.constraints:
                curve_eval = np.clip(curve_eval, val_min, val_max)
            if self.boundary and not control:
                curve_eval[0, :] = val_0
                curve_eval[-1, :] = val_f
            curve = self.fit_data(curve_eval.T)

        return curve

    def compute_velocity_traj(self, path_curves):
        # v(t) = dr/dt
        velocity = self._get_derivatives(path_curves)
        velocity = self._enforce_constraints(
            velocity, self.drone.x_min[3:6], self.drone.x_max[3:6], self.drone.x0[3:6], self.drone.xf[3:6])
        return velocity

    def compute_acceleration_traj(self, velocity_curves):
        # a(t) = d2r/dt2
        return self._get_derivatives(velocity_curves)

    def compute_force_traj(self, acceleration_curve):
        # f_des = m*(a + g_z)
        f_val = []
        for acceleration_i, acceleration_g in zip(acceleration_curve, np.array([0, 0, self.drone.g])):
            f_val.append(self.drone.mass *
                         (acceleration_i(self.t_seg_guess) + acceleration_g))
        force_curve = self.fit_data(f_val)

        return force_curve

    def compute_quaternion_traj(self, force_curves, psi_des=0.0):
        # based on conversions used in quaternion-based position controller
        # get normalized force vectors
        force_L_eval_np = self._eval_curves_np(
            self.t_seg_guess, force_curves)
        force_L_des_normed = (
            force_L_eval_np.T/np.linalg.norm(force_L_eval_np, 2, axis=1))
        # force in body frame
        force_b_des_normed = np.array([0, 0, 1])

        # calculate desired attitude quaternion
        q_des = np.zeros([force_L_des_normed.shape[1], 4])
        for i, force_L in enumerate(force_L_des_normed.T):
            q1 = 1/np.sqrt(2*(1+force_b_des_normed@force_L))*np.concatenate(
                [[1+force_b_des_normed@force_L], np.cross(force_b_des_normed, force_L)])
            q2 = np.array([np.cos(psi_des/2), 0, 0, np.sin(psi_des/2)])
            q_des[i, :] = quat_mult(q1, q2)
            q_des[i, :] = q_des[i, :]/np.linalg.norm(q_des[i, :])
        q_des_curve = self.fit_data(q_des.T)

        return q_des_curve

    def compute_angular_rate_traj(self, quaternion_curves):
        # omega = 2*Gamma(q)*dq/dt

        # get quaternion derivative and evaluate curves
        quaternion_np = self._eval_curves_np(
            self.t_seg_guess, quaternion_curves)
        quaternion_derivative_np = self._eval_curves_np(
            self.t_seg_guess, self._get_derivatives(quaternion_curves))

        # calculate angular rate
        omega = []
        for q, dq in zip(quaternion_np, quaternion_derivative_np):
            omega.append(2*quat_dyn_matrix(q)@dq)
        omega_curve = self.fit_data(np.array(omega).T)

        omega_curve = self._enforce_constraints(
            omega_curve, self.drone.x_min[10:], self.drone.x_max[10:], self.drone.x0[10:], self.drone.xf[10:])

        return omega_curve

    def compute_thrust_traj(self, force_L, quaternion):
        # evaluate quaternion and force
        quaternion_eval = self._eval_curves_np(self.t_seg_guess, quaternion)
        force_eval = self._eval_curves_np(self.t_seg_guess, force_L)

        # calculate desired thrust
        thrust_b_des = []
        for f, q in zip(force_eval, quaternion_eval):
            # desired thrust
            thrust_temp = (quat_mult(
                quat_mult(quat_inv(q), np.concatenate([[0], f])), q))[-1]
            # enforce limits
            thrust_b_des.append(thrust_temp)
        thrust_curve = self.fit_data(np.array([thrust_b_des]))
        thrust_curve = self._enforce_constraints(
            [thrust_curve], self.drone.u_min[0], self.drone.u_max[0], self.drone.uf[0], self.drone.uf[0], control=False)
        if not (self.boundary or self.constraints):
            thrust_curve = thrust_curve[0]
        return thrust_curve

    def compute_torque_traj(self, angular_rate):
        # tau = J*d omega/dx + omega × J*omega Eq. (2.4)
        angular_acceleration_np = self._eval_curves_np(
            self.t_seg_guess, self._get_derivatives(angular_rate))
        angular_rate_np = self._eval_curves_np(self.t_seg_guess, angular_rate)
        inertia_matrix = np.diag([self.drone.Ix, self.drone.Iy, self.drone.Iz])

        # calculate desired torque
        torque = []
        for rate, acc in zip(angular_rate_np, angular_acceleration_np):
            torque.append(inertia_matrix@acc +
                          np.cross(rate,inertia_matrix@rate))
        torque_curve = self.fit_data(np.array(torque).T)
        torque_curve = self._enforce_constraints(
            torque_curve, self.drone.u_min[1:], self.drone.u_max[1:], self.drone.uf[1:], self.drone.uf[1:], control=False)
        return torque_curve

    def take_guess(self, guess_level='position', guess_control=False, t_seg=None):
        """
        Generates a initial guess of state and control for the drone trajectory planning problem.

        Parameters:
            guess_level (str, optional): The level of guess. Defaults to 'position'.
                For 'velocity', it guesses the position and velocity of the drone.
                For 'orientation', it guesses the position, velocity and orientation of the drone.
                For 'time', it guesses only the time grid based on the boundary conditions and velocity constraints.
                If 'none', the guess is set according to terminal values.
            guess_control (bool, optional): Whether to guess the control. Defaults to False. 
                If False, the guess is set according to hover state control.
            t_seg (NoneType, optional): The time in segments. Defaults to None.

        Returns:
            None
        """
        
        self.guess_level = guess_level
        self.guess_control = guess_control
        
        self.compute_time_grid(t_seg)

        # State guess if guess level is low
        terminal_state = np.zeros((self.nx, len(self.t_seg_guess)))
        for i, elem in enumerate(self.drone.xf):
            terminal_state[i, :] = elem

        # Simple control guess
        terminal_control = np.zeros((self.nu, len(self.t_seg_guess)))
        for i, elem in enumerate(self.drone.uf):
            terminal_control[i, :] = elem

        if self.path_dim == 2:  # Compute the drone's position at each time point in t_seg
            z_guess_seg = np.array([self.drone_position(t, self.drone.x0[2], self.drone.xf[2],
                                   self.t_seg_guess[0], self.t_seg_guess[-1]) for t in self.t_seg_guess])
            self.path_waypoints = np.column_stack(
                (self.path_waypoints, z_guess_seg))

        # make full guess
        if self.guess_level == 'none' or self.guess_level == 'time':
            # interpolate the path based on guess of time grid
            path = np.zeros((len(self.t_seg_guess),3))
            for  i, t in enumerate(self.t_seg_guess):
                path[i,:] = self.drone_position(t, self.drone.x0[:3], self.drone.xf[:3],self.t_seg_guess[0],self.t_seg_guess[-1])
            position = self.compute_position_traj(path)
        else:
            position = self.compute_position_traj(self.path_waypoints)
        velocity = self.compute_velocity_traj(position)
        acceleration = self.compute_acceleration_traj(velocity)
        force = self.compute_force_traj(acceleration)
        orientation = self.compute_quaternion_traj(force)
        angular_rate = self.compute_angular_rate_traj(orientation)

        # guess control
        if guess_control:
            thrust = self.compute_thrust_traj(force, orientation)
            torque = self.compute_torque_traj(angular_rate)
        else:
            thrust = self.fit_data(np.array([terminal_control[0, :]]))
            torque = self.fit_data(terminal_control[1:, :])
        control_curve = []
        control_curve.append(thrust)
        control_curve.extend(torque)

        if guess_level in ['position', 'time', 'none']:
            velocity = self.fit_data(terminal_state[3:6, :])
            orientation = self.fit_data(terminal_state[6:10, :])
            angular_rate = self.fit_data(terminal_state[10:, :])
        elif guess_level == 'velocity':
            orientation = self.fit_data(terminal_state[6:10, :])
            angular_rate = self.fit_data(terminal_state[10:, :])
        elif guess_level == 'orientation':
            angular_rate = self.fit_data(terminal_state[10:, :])
        elif guess_level == 'angular_rate':
            pass

        state_curve = position + velocity + orientation + angular_rate
        
        # sample the guess
        state_grid = []
        control_grid = []
        if self.single_segment:
            # single segment evaluation in collocation points
            # self.n_col = np.array([int(np.sum(self.n_col)-self.n_seg)+1]) # for the same number of collocation points
            self.n_col = np.array([self.n_col[0]])
            self.n_seg = 1
            time = []
            grid, __, __ = self.col_points(
                int(self.n_col[0]), [self.t_seg_guess[0], self.t_seg_guess[-1]])
            time.append(grid)
            self.time_guess = time
            state_grid_temp = self._eval_curves_np(time[0], state_curve)
            state_grid.append(state_grid_temp.T)
            control_grid_temp = self._eval_curves_np(time[0], control_curve)
            control_grid.append(control_grid_temp.T)
        else:
            # multiple segments evaluation in collocation points
            for time in self.time_guess:
                state_grid_temp = self._eval_curves_np(time, state_curve)
                state_grid.append(state_grid_temp.T)
                control_grid_temp = self._eval_curves_np(time, control_curve)
                control_grid.append(control_grid_temp.T)

        self.state_guess = state_grid
        self.control_guess = control_grid

    def plot(self):
        fig1 = plt.figure(figsize=(8, 12))
        ax = fig1.add_subplot(211, projection='3d')
        ax.plot(self.path_waypoints[:, 0], self.path_waypoints[:, 1], self.path_waypoints[:, 2], 'o-')
        ax.set_title('Waypoints')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        
        ax = fig1.add_subplot(212, projection='3d')
        for i, (state_guess_seg, t_seg) in enumerate(zip(self.state_guess, self.time_guess)):
            ax.plot(state_guess_seg[0, :], state_guess_seg[1, :],state_guess_seg[2, :], 'o-')
            # ax.plot(self.path_waypoints[:, 0], self.path_waypoints[:, 1], self.path_waypoints[:, 2], 'x')
            ax.plot(self.path_waypoints[:, 0], self.path_waypoints[:, 1], self.path_waypoints[:, 2], 'x')
        ax.set_title('Waypoints')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        fig2 = plt.figure(figsize=(8, 12))
        ax = fig2.add_subplot(211)
        for i, (state_guess_seg, t_seg) in enumerate(zip(self.state_guess, self.time_guess)):
            ax.plot(t_seg, state_guess_seg[0, :])#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[1, :])#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[2, :])#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Position')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        # ax.legend()
        
        ax = fig2.add_subplot(212)
        for i, (state_guess_seg, t_seg) in enumerate(zip(self.state_guess, self.time_guess)):
            ax.plot(t_seg, state_guess_seg[3, :])#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[4, :])#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[5, :])#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Velocity')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        # ax.legend()
        
        fig3 = plt.figure(figsize=(8, 12))
        ax = fig3.add_subplot(211)
        for i, (state_guess_seg, t_seg) in enumerate(zip(self.state_guess, self.time_guess)):
            ax.plot(t_seg, state_guess_seg[7, :])#, label='q_x')#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[8, :])#, label='q_y')#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[9, :])#, label='q_z')#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Orientation')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Quaternion')
        # ax.legend()
        
        ax = fig3.add_subplot(212)
        for i, (state_guess_seg, t_seg) in enumerate(zip(self.state_guess, self.time_guess)):
            ax.plot(t_seg, state_guess_seg[10, :])#, label=f'\omega_x')#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[11, :])#, label=f'\omega_y')#, label=f'Segment {i+1}')
            ax.plot(t_seg, state_guess_seg[12, :])#, label=f'\omega_z')#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Angular Rate')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angular Rate [rad/s]')
        # ax.legend()
        
        fig4 = plt.figure(figsize=(8, 12))
        ax = fig4.add_subplot(111)
        for i, (control_guess_seg, t_seg) in enumerate(zip(self.control_guess, self.time_guess)):
            ax.plot(t_seg, control_guess_seg[0, :])#, label='q_x')#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Thrust')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Thrust [N]')
        
        fig4 = plt.figure(figsize=(8, 12))
        ax = fig4.add_subplot(111)
        for i, (control_guess_seg, t_seg) in enumerate(zip(self.control_guess, self.time_guess)):
            ax.plot(t_seg, control_guess_seg[1, :])#, label='q_y')#, label=f'Segment {i+1}')
            ax.plot(t_seg, control_guess_seg[2, :])#, label='q_z')#, label=f'Segment {i+1}')
            ax.plot(t_seg, control_guess_seg[3, :])#, label='q_z')#, label=f'Segment {i+1}')
        ax.set_title('Initial Guess of Torque')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [N⋅m]')
        # ax.legend()
        
        
        plt.show()
        
    def save(self, folder = None, file_name = "default"):
        """ Save initial guess nodes to csv file

        Args:
            folder (str, optional): name of the folder for saving. Defaults to None.
            file_name (str, optional): save csv as "ps_init_"+"file_name". Defaults to "default".
        """
        if self.state_guess is None or self.control_guess is None:
            Warning("No initial guess is provided. Please provide an initial guess.")
            return
        
        import os, csv, sys
        if folder is None:
            folder = os.path.dirname(sys.argv[0])
        file_path = os.path.join(folder,"data_traj",'ps_init_'+file_name+'.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(["Path", "Segmentation",
                            "State", "Control", "Time"])
            writer.writerow([self.path_waypoints, self.n_col,
                            self.state_guess, self.control_guess, self.time_guess])

    def simulate_dynamics(self, state, control, dt=1e-3):
        """
        Method for simulating the dynamics of the drone 
        given the current state and control vectors and the time step dt.

        Parameters
        ----------
        state: numpy.ndarray
            Current state vector of the drone.
        control: numpy.ndarray
            Current control vector of the drone.
        dt: float
            The time step for simulation.

        Returns
        -------
        numpy.ndarray
            The updated state vector after one time step.
        """
        
        updated_state = state + dt*self.drone.dynamics(state, control)
        return updated_state


def example(world = 'simple'):
    import os, sys
    
    current_folder = os.path.dirname(sys.argv[0])
    
    if world == 'gen':
        # np.random.seed(42)
        # Generate random values for the first two columns (-3 to 3)
        col1 = np.random.uniform(-3, 3, size=(10, 1))
        col2 = np.random.uniform(-3, 3, size=(10, 1))

        # Generate random values for the third column (0 to 2)
        col3 = np.random.uniform(0, 2, size=(10, 1))

        # Concatenate the columns to form the final matrix
        path_waypoints = np.concatenate((col1, col2, col3), axis=1)
    else:
        file_name = 'lt_'+world+'.npz'
        file_path = os.path.join(current_folder,'data_path', 'lt_'+world+'.npz')
        lt_path = np.load(file_path, allow_pickle=True)
        path_waypoints = lt_path['path']
    
    ps_init = PsInit(path_waypoints, interpolation_method='spline',
                     constraints=True, boundary=True, single_segment = False, p_min = 20)
    ps_init.take_guess(guess_level='angular_rate', guess_control=True)
    # ps_init.save()
    ps_init.plot()

if __name__ == '__main__':
    # example('gen')
    example('orchard')
