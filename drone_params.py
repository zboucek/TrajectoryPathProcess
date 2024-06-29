import numpy as np
import sympy as sym


class Drone:
    """Parameters of full drone model controlled with collective thrust and body torques.
        Attitude of drone is represented by unit quaternion.
        Parameters are based on the papers:
        Hummingbird:
            Z. Bouček and M. Flídr, "Explicit Interpolating Control of Unmanned Aerial Vehicle,"
            2019 24th International Conference on Methods and Models in Automation and Robotics (MMAR),
            2019, pp. 384-389, doi: 10.1109/MMAR.2019.8864719.
        Crazyflie:
            Förster, Julian. “System Identification of the Crazyflie 2.0 Nano Quadrocopter.” (2015).
            Landry, B. (2015). Planning and Control for Quadrotor Flight through Cluttered Environments. 
                Thesis, 2014, 71. http://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf
    """

    def __init__(self, drone_type: str = 'crazyflie', aerodynics: bool = True):
        # drone parameters
        self.g = 9.81305
        self.aerodynamics = aerodynics
        if drone_type == 'crazyflie':
            # Bitcraze Crazyflie
            self.mass = 0.032
            # Matrix of Inertia
            # # according to Landry, B. (2015). Planning and Control for Quadrotor Flight through Cluttered Environments.
            # #   Thesis, 2014, 71. http://groups.csail.mit.edu/robotics-center/public_papers/Landry15.pdf
            # self.Ix = 2.3951e-5
            # self.Iy = 2.3951e-5
            # self.Iz = 3.2347e-5
            # according to Förster thesis
            self.Ix = 6.410179e-6
            self.Iy = 6.410179e-6
            self.Iz = 9.860228e-6
            self.l = 0.0397
            self.proppeler_d = 51*1e-3  # propeller diameter in meters
            area_x = 2.0
            area_y = 2.0
            area_z = 2.0
            self.speed_limit = 2.0
            self.quat_limit = 1.0
            self.arate_limit = np.Inf
            # aerodynamic effect
            # F_a = K_aero*T*R'*v
            # F = R*([0 0 T]+F_a)
            self.K_aero = 1e-7*np.array([[-10.2506, -0.3177, -0.4332],
                                    [-0.3177, -10.2506, -0.4332],
                                    [-7.7050, -7.7050, -7.5530]])
        else:
            # AscTec Hummingbird
            self.mass = 0.542
            self.Ix = 0.0064
            self.Iy = 0.0064
            self.Iz = 0.0125
            self.l = 0.174
            self.proppeler_d = 0
            area_x = 10.0
            area_y = 10.0
            area_z = 10.0
            self.speed_limit = 5.0
            self.quat_limit = 1.0
            self.arate_limit = 5.0
            self.K_aero = np.zeros((3, 3))

        self.safe_radius = (self.l + self.proppeler_d/2)*1.1
        self.area = np.array([area_x, area_y, area_z])

        # weight matrices (same for every axis)
        if drone_type == 'crazyflie':
            Qpos = 1/(0.5*self.area)**2
            Qspeed = 1/(0.5*self.speed_limit)**2
            Qquat = 100  # quaternion
            Qarate = 1/np.deg2rad(100)**2  # angular rate
            Rthrust = 1/((4*0.15)-self.mass*self.g)**2
            tau_max = 0.15*self.l - 0*self.l
            Rtorque_x = 1/tau_max**2
            Rtorque_y = 1/tau_max**2
            Rtorque_z = 1/(2*(0.005964552*0.15 + 1.563383e-5))**2
            Rtorque = np.array([Rtorque_x, Rtorque_y, Rtorque_z])
        else:
            Qpos = 5000*np.ones(3)  # postition
            Qspeed = 800  # speed
            Qquat = 100  # quaternion
            Qarate = 10  # angular rate
            Rthrust = 100   # collective thrust
            Rtorque = 10*np.ones(3)    # body torque
        self.Q = np.diag(np.block([Qpos, Qspeed *
                                   np.ones(3), Qquat*np.ones(4), Qarate*np.ones(3)]))
        self.R = np.diag(np.block([Rthrust, Rtorque]))

        self.nx = self.Q.shape[1]  # dimension of state
        self.nu = self.R.shape[1]  # dimension of control

        # state and control constraints
        # area = int(10/2)
        self.x_max = np.block([self.area, self.speed_limit*np.ones(3),
                               self.quat_limit*np.ones(4), self.arate_limit*np.ones(3)])
        self.x_min = -self.x_max.copy()
        self.x_min[2] = 0
        # set q_w > 0 to lock the attitude representation and avoid flipping between positive and negative rotations
        self.x_min[6] = 0
        # x_max[2] = 10
        if drone_type == 'crazyflie':
            self.u_max = np.block(
                [4*0.15, tau_max, tau_max,  2*(0.005964552*0.15 + 1.563383e-5)])
            self.u_min = -self.u_max.copy()
            self.u_min[0] = 0.0
        else:
            self.u_max = np.block([18.4167, 2.8543*np.ones(3)])
            self.u_min = -self.u_max.copy()
            self.u_min[0] = 0.2051

        self.uf = np.zeros(self.nu)
        self.uf[0] = self.mass*self.g

        # time constraints
        self.t0 = 0.0
        self.tf_min = 0.0
        if drone_type == 'crazyflie':
            self.tf_max = 3.0*60.0
        else:
            self.tf_max = 15*60.0

        # boundary constraints
        self.x0 = np.zeros(self.nx)
        self.x0[0] = -.8*self.area[0]  # x
        self.x0[1] = -.8*self.area[1]  # y
        self.x0[2] = -.8*self.area[2]+self.x_max[2]  # z
        self.x0[6] = 1.0  # q_w
        self.xf = np.zeros(self.nx)
        self.xf[0] = .8*self.area[0]  # x
        self.xf[1] = .8*self.area[1]  # y
        self.xf[2] = -.8*self.area[2]+self.x_max[2]  # z
        self.xf[6] = 1.0  # q_w
        
        state = sym.symbols('x0:13')  
        control = sym.symbols('u0:4')
        dynamics = self.__symbolic_dynamics(state, control)
        self.eval_dynamics = sym.lambdify((state, control), dynamics)
        self.eval_dynamics_state = []
        for state_dyn in dynamics:
            self.eval_dynamics_state.append(sym.lambdify((state, control), state_dyn))

    def __eval_dynamics_old(self, state_t, control_t):
        """
        Dynamics evaluation based on calculations in ps_build_dynamics

            state vector state_t:
                0: rx, 1: ry, 2: rz, 3: vx, 4: vy, 5: vz, 6: qw, 7: qx, 8: qy, 9: qz, 10: wx, 11: wy, 12: wz
            control vector control_t:
                0: T, 1: taux, 2: tauy, 3: tauz
        """

        return np.array([state_t[3], state_t[4], state_t[5],
                         2*(state_t[7]*state_t[9]+state_t[8] *
                            state_t[6])*control_t[0]/(self.mass*((state_t[6]**2+state_t[7]**2+state_t[8]
                                                                  ** 2+state_t[9]**2))),
                         2*(state_t[8]*state_t[9]-state_t[7] *
                            state_t[6])*control_t[0]/(self.mass*((state_t[6]**2+state_t[7]**2+state_t[8]
                                                                  ** 2+state_t[9]**2))),
                         -self.g + control_t[0]*(-2*(state_t[7]**2+state_t[8]
                                                     ** 2)/(state_t[6]**2+state_t[7]**2+state_t[8]
                                                 ** 2+state_t[9]**2)+1)/self.mass,
                         - (state_t[6]*state_t[12])/2 - (state_t[8] *
                                                         state_t[10])/2 - (state_t[9]*state_t[12])/2,
                         (state_t[6]*state_t[10])/2 + (state_t[8] *
                                                       state_t[12])/2 - (state_t[9]*state_t[11])/2,
                         (state_t[6]*state_t[11])/2 - (state_t[7] *
                                                       state_t[12])/2 + (state_t[9]*state_t[10])/2,
                         (state_t[6]*state_t[12])/2 + (state_t[7] *
                                                       state_t[11])/2 - (state_t[8]*state_t[10])/2,
                         (control_t[1] + (self.Iy - self.Iz)
                          * state_t[11]*state_t[12])/self.Ix,
                         (control_t[2] + (self.Iz - self.Ix)
                          * state_t[10]*state_t[12])/self.Iy,
                         (control_t[3] + (self.Ix - self.Iy)*state_t[10]*state_t[11])/self.Iz])

    def symbolic_dynamics(self):
        """This function calculates quadrotor dynamics using sympy."""

        # state/control variables and parameters
        qw, qx, qy, qz = sym.symbols('qw, qx, qy, qz', real=True)
        wx, wy, wz = sym.symbols('wx, wy, wz', real=True)
        x, y, z = sym.symbols('x, y, z', real=True)
        vx, vy, vz = sym.symbols('vx, vy, vz', real=True)
        m = sym.Symbol('m', positive=True)
        g = sym.Symbol('g', positive=True)
        Ix, Iy, Iz = sym.symbols('Ix, Iy, Iz', positive=True)
        T = sym.Symbol('T', positive=True)
        taux, tauy, tauz = sym.symbols('taux, tauy, tauz', real=True)

        # vectors
        u = sym.Matrix([T, taux, tauy, tauz])
        q = sym.Matrix([qw, qx, qy, qz])
        qv = sym.Matrix(q[1:])
        w = sym.Matrix([wx, wy, wz])
        r = sym.Matrix([x, y, z])
        v = sym.Matrix([vx, vy, vz])
        I = sym.diag(Ix, Iy, Iz)
        x = sym.Matrix([r, v, q, w])
        tau = sym.Matrix(u[1:])

        # rotation matrix
        R = sym.Matrix([[qw**2 + qx**2 - qy**2 - qz**2,         2*qx*qy - 2*qw*qz,         2*qw*qy + 2*qx*qz],
                        [2*qw*qz + 2*qx*qy, qw**2 - qx**2 + qy **
                            2 - qz**2,         2*qy*qz - 2*qw*qx],
                        [2*qx*qz - 2*qw*qy,         2*qw*qx + 2*qy*qz, qw**2 - qx**2 - qy**2 + qz**2]])

        Ralt2 = R*sym.sqrt(q.dot(q))**(-2)

        # alternative rotation matrix
        s = sym.sqrt(q.dot(q))**(-2)
        Ralt = sym.Matrix([[1-2*s*(q[2]**2+q[3]**2), 2*s*(q[1]*q[2]-q[3]*q[0]), 2*s*(q[1]*q[3]+q[2]*q[0])],
                           [2*s*(q[1]*q[2]+q[3]*q[0]), 1-2*s *
                            (q[1]**2+q[3]**2), 2*s*(q[2]*q[3]-q[1]*q[0])],
                           [2*s*(q[1]*q[3]-q[2]*q[0]), 2*s*(q[2]*q[3]+q[1]*q[0]), 1-2*s*(q[1]**2+q[2]**2)]])

        # gamma matrix for rotational dynamics
        Omega = sym.Matrix([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
        Gamma = sym.Matrix.hstack(-qv, qw*sym.eye(3)-Omega)

        # equations of motion
        dr = v
        dvalt2 = -sym.Matrix([0, 0, g]) + Ralt2*sym.Matrix([0, 0, T])/m
        dvalt = -sym.Matrix([0, 0, g]) + Ralt*sym.Matrix([0, 0, T])/m
        dv = -sym.Matrix([0, 0, g]) + R*sym.Matrix([0, 0, T])/m
        dw = -I.inv() * (w.cross(I*w)) + I.inv()*tau
        dq = 0.5*Gamma.T*w

        # print(dvalt2)
        # print(dvalt)
        # print('---')
        # print(dv)
        # print(dq)
        # print(dw)
        return [dr,dvalt, dq, dw]
    
    def __symbolic_dynamics(self, state, control):
        """Symbolic dynamics for injection to ps_solution evaluation and OCP definition in ps_control"""
        
        # Unpack vectors
        qw, qx, qy, qz = state[6:10]
        q = sym.Matrix([qw, qx, qy, qz])
        qv = sym.Matrix(q[1:])
        w = sym.Matrix(state[10:])
        v = sym.Matrix(state[3:6:])
        thrust = control[0]
        torque = sym.Matrix(control[1:])
        I = np.diag([self.Ix, self.Iy, self.Iz])
        
        # get rotational matrix
        s = sym.sqrt(q.dot(q))**(-2)
        R = sym.Matrix([[1-2*s*(q[2]**2+q[3]**2), 2*s*(q[1]*q[2]-q[3]*q[0]), 2*s*(q[1]*q[3]+q[2]*q[0])],
                           [2*s*(q[1]*q[2]+q[3]*q[0]), 1-2*s *
                            (q[1]**2+q[3]**2), 2*s*(q[2]*q[3]-q[1]*q[0])],
                           [2*s*(q[1]*q[3]-q[2]*q[0]), 2*s*(q[2]*q[3]+q[1]*q[0]), 1-2*s*(q[1]**2+q[2]**2)]])
        
        # gamma matrix for rotational dynamics
        Omega = sym.Matrix([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
        Gamma = sym.Matrix.hstack(-qv, qw*sym.eye(3)-Omega)

        # equations of motion
        dr = v
        if self.aerodynamics:
            F_a = self.K_aero*thrust*R.T*v
        else:
            F_a = np.zeros(3)
        dv = -sym.Matrix([0, 0, self.g]) + (R*(sym.Matrix([0, 0, thrust])+F_a))/self.mass
        invI = np.linalg.pinv(I)
        dq = 0.5*Gamma.T*w
        dw = -invI * (w.cross(I*w)) + invI*torque
        
        return sym.Matrix([dr, dv, dq, dw])[:]


class AttitudeController:
    """
    Attitude controller class.
    """

    def __init__(self, k_quat, k_rate, inertia_matrix):
        """
        Initialize the attitude controller.

        Args:
            k_quat (float): Quaternion gain.
            k_rate (float): Angular velocity gain.
            inertia_matrix (ndarray): Inertia matrix of drone.
        """
        self.k_quat = k_quat
        self.k_rate = k_rate
        self.inertia_matrix = inertia_matrix
        self.prev_err = np.array([1, 0, 0, 0])
        self.prev_time = 0.0

    def update(self, quat_des, quat_cur, ang_vel_cur, ang_vel_des, curr_time):
        """
        Calculates the control torque for the attitude controller.

        Args:
            quat_des (ndarray): Desired quaternion.
            quat_cur (ndarray): Current quaternion.
            ang_vel_cur (ndarray): Current angular velocity.
            ang_vel_des (ndarray): Desired angular velocity.
            curr_time (float): Current time.

        Returns:
            ndarray: Control torque.
        """

        # Calculate quaternion error
        q_err = quat_err(quat_cur, quat_des)

        # Calculate derivative of quaternion error
        d_err = quat_mult(0.5*q_err, np.concatenate([[0], ang_vel_cur]) - quat_mult(
            quat_mult(quat_inv(q_err), np.concatenate([[0], ang_vel_des])), q_err))
        norm = np.linalg.norm(d_err, 2)
        if norm == 0.0:
            d_err = np.zeros(4)
            d_err[0] = 1
        else:
            d_err /= norm
        d_err = (d_err - self.prev_err)/(curr_time - self.prev_time)
        norm = np.linalg.norm(d_err, 2)
        if norm == 0.0:
            d_err = np.zeros(4)
            d_err[0] = 1
        else:
            d_err /= norm

        # Update previous error and time
        self.prev_err = d_err
        self.prev_time = curr_time

        # Calculate control torque
        torque = self.inertia_matrix@quat_dyn_matrix(
            q_err) @ (-2*self.k_quat@self.k_rate@d_err - self.k_rate**2 @ (q_err - np.array([1, 0, 0, 0])))
        torque += np.cross(ang_vel_cur, self.inertia_matrix@ang_vel_cur)

        return torque


class PositionController:
    """
    Position controller class.
    """

    def __init__(self, k_position, k_speed, mass, gravity):
        """
        Initialize the PositionController class.

        Args:
            k_position (np.ndarray): The position gain matrix.
            k_speed (np.ndarray): The speed gain matrix.
            mass (float): The mass of the drone.
            gravity (float): The gravitational acceleration.
        """
        self.k_position = k_position
        self.k_speed = k_speed
        self.mass = mass
        self.gravity = gravity

    def update(self, pos_des, pos_cur, vel_des, vel_cur):
        """
        Calculate the control force based on desired and current positions and velocities.

        Args:
            pos_des (np.ndarray): The desired position.
            pos_cur (np.ndarray): The current position.
            vel_des (np.ndarray): The desired velocity.
            vel_cur (np.ndarray): The current velocity.

        Returns:
            np.ndarray: The calculated control force.
        """

        force = self.mass * (
            self.k_position @ (pos_des - pos_cur)
            + self.k_speed @ (vel_des - vel_cur)
            + np.array([0, 0, self.gravity])
        )

        return force


def connect_controllers(pos_controller, att_controller, pos_des, pos_cur, vel_des, vel_cur, quat_cur, ang_vel_cur, ang_vel_des, psi_des, prev_quat_des, curr_time):
    # Calculate desired force in local frame
    force_L_des = pos_controller.update(pos_des, pos_cur, vel_des, vel_cur)
    force_L_des_normed = force_L_des/np.linalg.norm(force_L_des, 2)

    # Calculate desired attitude quaternion
    force_b_des_normed = np.array([0, 0, 1])
    q1 = 1/np.sqrt((2*(1+force_b_des_normed@force_L_des_normed)))*np.concatenate(
        [[1+force_b_des_normed@force_L_des_normed], np.cross(force_b_des_normed, force_L_des_normed)])
    q2 = np.array([np.cos(psi_des/2), 0, 0, np.sin(psi_des/2)])
    q_des_ = quat_mult(q1, q2)
    dot_prod = prev_quat_des@q_des_
    if dot_prod < 0:
        q_des = -q_des_/np.linalg.norm(q_des_, 2)
    else:
        q_des = q_des_/np.linalg.norm(q_des_, 2)


    # Update previous desired quaternion
    prev_quat_des = q_des
    
    
    # R = quat2rotmat(q_des)
    # force_b_des_ = (R)@force_L_des
    force_b_des_ = quat_mult(
        quat_mult(quat_inv(q_des), np.concatenate([[0], force_L_des])), q_des)
    

    return force_b_des_[-1], prev_quat_des


def quat_dyn_matrix(q):
    """Matrix for quaternion dynamics described in Appendix equation (A.18) in dissertation"""
    return np.array([[-q[1],  q[0], -q[3],  q[2]],
                    [-q[2],  q[3],  q[0], -q[1]],
                    [-q[3], -q[2],  q[1],  q[0]]])


def quat2rotmat(q):
    """
    Converts a quaternion to a rotation matrix.

    Parameters:
        q (numpy.ndarray): The quaternion to be converted.

    Returns:
        numpy.ndarray: The resulting rotation matrix.
    """
    s = np.sqrt(q.dot(q))**(-2)
    R = np.array([[1-2*s*(q[2]**2+q[3]**2), 2*s*(q[1]*q[2]-q[3]*q[0]), 2*s*(q[1]*q[3]+q[2]*q[0])],
                  [2*s*(q[1]*q[2]+q[3]*q[0]), 1-2*s *
                   (q[1]**2+q[3]**2), 2*s*(q[2]*q[3]-q[1]*q[0])],
                  [2*s*(q[1]*q[3]-q[2]*q[0]), 2*s*(q[2]*q[3]+q[1]*q[0]), 1-2*s*(q[1]**2+q[2]**2)]])
    return R


def quaternion2euler(q):
    """
    Convert a quaternion to Euler angles.

    Parameters:
        q (numpy.ndarray): A 1-D array representing the quaternion in the order (w, x, y, z).

    Returns:
        tuple: A tuple containing the roll, pitch, and yaw angles in degrees.
    """
    # Normalize the quaternion
    q = q / np.linalg.norm(q)

    # Extract the components of the quaternion
    w, x, y, z = q

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return roll, pitch, yaw


def quat_mult(q1, q2):
    """Return quaternion product of quaternion q1 and q2"""

    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]
    s = s1*s2 - v1@v2.T
    v = s1*v2 + s2*v1 + np.cross(v1, v2)

    return np.hstack([[s], v])


def quat_err(quat_cur, quat_des):
    # Calculate quaternion error
    q_err = quat_mult(quat_inv(quat_des), quat_cur)
    if q_err[0] < 0:
        q_err = -q_err
    return q_err/np.linalg.norm(q_err, 2)


def quat_distance(q1, q2):
    """
    Calculate the distance between two quaternions.

    Parameters:
        q1 (numpy.ndarray): The first quaternion.
        q2 (numpy.ndarray): The second quaternion.

    Returns:
        float: The distance between the two quaternions.
    """
    q1 = q1/np.linalg.norm(q1)
    q2 = q2/np.linalg.norm(q2)
    dot_product = np.dot(q1, q2)
    return abs(1 - dot_product)


def quat_distance_angle(q1, q2):
    """
    Calculate the angle between two quaternions.

    Parameters:
        q1 (array-like): The first quaternion.
        q2 (array-like): The second quaternion.

    Returns:
        float: The angle between the two quaternions.
    """
    dot_product = np.dot(q1, q2)
    angle = 2 * np.arccos(abs(dot_product))
    return angle


def quat_inv(q):
    """Return inverse quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def example():
    drone = Drone()
    k_quat = 0.7*np.eye(4)
    k_rate = 3.5*np.eye(4)
    k_pos_lat = 0.2
    k_speed_lat = 7.0
    k_position = np.diag([k_pos_lat, k_pos_lat, 1.0])
    k_speed = np.diag([k_speed_lat, k_speed_lat, 0.8])
    att_controller = AttitudeController(
        k_quat, k_rate, np.diag([drone.Ix, drone.Iy, drone.Iz]))
    pos_controller = PositionController(
        k_position, k_speed, drone.mass, drone.g)
    pos_des = drone.xf[:3]
    pos_cur = drone.x0[:3]
    # pos_des[1:] = pos_cur[1:]
    vel_des = drone.xf[3:6]
    vel_cur = drone.x0[3:6]
    quat_des = drone.xf[6:10]
    quat_cur = drone.x0[6:10]
    # quat_cur = np.array([0.3, 0.8, 0.001, 0.002])
    quat_cur = np.array([1,0,0,0])
    quat_cur = quat_cur/np.linalg.norm(quat_cur, 2)
    ang_vel_cur = drone.x0[10:13]
    ang_vel_des = drone.xf[10:13]
    psi_des = 0.0
    prev_quat_des = quat_des
    curr_time = 0.01
    t_sample = 0.1
    t_sample_fast = 0.01
    t_max = 120
    N_steps = int(t_max/t_sample)
    state = np.concatenate([pos_cur, vel_cur, quat_cur, ang_vel_cur])
    time = np.linspace(t_sample, t_max, N_steps)
    state_mem = np.zeros([13, N_steps])
    euler_mem = np.zeros([3, N_steps])
    for i in range(N_steps):
        pos_cur, vel_cur, quat_cur, ang_vel_cur = np.split(state, [3, 6, 10])
        # torque = att_controller.update(np.array([1,0,0,0]), quat_cur, ang_vel_cur, [0,0,0],curr_time = curr_time)
        thrust, quat_des = connect_controllers(
            pos_controller, att_controller, pos_des, pos_cur, vel_des, vel_cur, quat_cur, ang_vel_cur, ang_vel_des, psi_des, prev_quat_des, curr_time)
        # print(thrust, torque)
        # thrust = 1.1*drone.mass*drone.g
        for iter in range(int(round(t_sample/t_sample_fast))):
            pos_cur, vel_cur, quat_cur, ang_vel_cur = np.split(state, [3, 6, 10])
            # Calculate control torque
            torque = att_controller.update(
                quat_des, quat_cur, ang_vel_cur, ang_vel_des, curr_time+iter*t_sample_fast)
            control = np.concatenate([[thrust], torque])
            # control = np.clip(control, drone.u_min, drone.u_max)

            dstate = np.array(drone.eval_dynamics(state, control))
            state += t_sample_fast*dstate
            state[6:10] = state[6:10]/np.linalg.norm(state[6:10], 2)
        euler_mem[:, i] = quaternion2euler(state[6:10])
        # state = np.clip(state, drone.x_min, drone.x_max)
        state_mem[:, i] = state
        curr_time += t_sample
        print(np.linalg.norm(pos_cur-pos_des))

    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(0, 3):
        plt.plot(time, state_mem[i, :])
    plt.xlabel('t[s]')
    plt.ylabel('$r[m]$')
    plt.legend(['$r_x$', '$r_y$', '$r_z$'])

    plt.figure()
    for i in range(6, 10):
        plt.plot(time, state_mem[i, :])
    plt.xlabel('t[s]')
    plt.ylabel('$q$')
    plt.legend(['$q_w$', '$q_x$', '$q_y$', '$q_z$'])

    plt.figure()
    for i in range(0, 3):
        plt.plot(time, euler_mem[i, :])
    plt.xlabel('t[s]')
    plt.ylabel('$\phi[deg]$')
    plt.legend(['$\phi$', '$\theta$', '$\psi$'])
    plt.show()


if __name__ == '__main__':
    
    example()
    # drone = Drone()
    # Qdiag = np.diag(drone.Q)
    # Qxq = np.diag(np.block([Qdiag[:6],Qdiag[-3:]]))
    # print(Qdiag)
    # print(Qxq)
    # print(drone.R)
    # print(drone.uf)