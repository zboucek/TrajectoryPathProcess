import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tplt
from pyomo.environ import *
from worlds import Gridmap, Tree
import cloudpickle
from drone_params import Drone
from drone_params import quat_distance
from ps_solution import PsSolution
from pyomo.core.plugins.transform.scaling import ScaleModel


class PsControl:
    """The PsControl solves drone trajectory planning problem in manner of Optimal Control Problem (OCP).
    OCP is transcribed using Multi-Segment Pseudospectal Collocation Method.
    Whole problem is modeled in Pyomo optimization toolbox and solved using NPL solver IPOPT.
    The problem is discretized using Chebyshev polynomial approximation on Gauss-Lobatto grid points.
    """

    def ps_build_dynamics(self, m, drone):
        """Hardcoded dynamics for multirotor drone with attitude quaternion representation.
        state vector m.x:
            0: rx, 1: ry, 2: rz, 3: vx, 4: vy, 5: vz, 6: qw, 7: qx, 8: qy, 9: qz, 10: wx, 11: wy, 12: wz
        control vector m.u:
            0: T, 1: taux, 2: tauy, 3: tauz
        """

        # Get symbolic dynamics function
        m.f_dx = drone.eval_dynamics_state

        def f_unit_q_definition(m, i):
            return 1.0 == sqrt(
                (m.x[6, i] ** 2 + m.x[7, i] ** 2 + m.x[8, i] ** 2 + m.x[9, i] ** 2)
            )

        m.unit_quaternion = Constraint(m.cols, rule=f_unit_q_definition)

        def dynamic_constraint(m, i, idx_state, i_seg, N_seg, seg_start, f_dyn):
            # general constraint for approximation of derivative with chebyshev differential matrix
            return 0.0 == ((m.knot_time[i_seg + 1] - m.knot_time[i_seg]) / 2) * (
                f_dyn[idx_state](m.x[:, seg_start + i], m.u[:, seg_start + i])
            ) - sum(
                m.x[idx_state, seg_start + k] * m.cheby[i_seg * 3 + 2][i, k]
                for k in range(N_seg)
            )

        m.dynamics_const = ConstraintList()
        for seg, n in enumerate(m.n_col):
            seg_start = m.n_col_add[seg]
            for i in range(n):
                for j in m.nx:
                    # add constraints for dynamics
                    m.dynamics_const.add(
                        dynamic_constraint(m, i, j, seg, n, seg_start, m.f_dx)
                    )

        # equiality of derivatives in segment knot points
        if m.n_seg > 1:

            def dynamic_knot_constraint(m, i, idx_state, f_dyn):
                # general constraint for approximation of derivative with chebyshev differential matrix
                return 0.0 == (
                    f_dyn[idx_state](m.x[:, i], m.u[:, i])
                    - f_dyn[idx_state](m.x[:, i - 1], m.u[:, i - 1])
                )

            m.dynamics_knot_const = ConstraintList()
            for i in m.n_col_add[1:]:
                for j in m.nx:
                    # add constraints for dynamics
                    m.dynamics_knot_const.add(dynamic_knot_constraint(m, i, j, m.f_dx))
            if m.knot_control_derivative:

                def control_derivative_knot_constraint(m, idx_seg1, idx_control):
                    # control derivative equility in knot points
                    idx_seg1 = idx_seg1
                    idx_seg2 = idx_seg1 + 1
                    N_seg1 = m.n_col[idx_seg1]
                    N_seg2 = m.n_col[idx_seg2]
                    seg1 = m.n_col_add[idx_seg1]
                    seg2 = m.n_col_add[idx_seg2]
                    return 0.0 == (
                        sum(
                            m.u[idx_control, seg1 + k]
                            * m.cheby[idx_seg1 * 3 + 2][N_seg1 - 1, k]
                            for k in range(N_seg1)
                        )
                        - sum(
                            m.u[idx_control, seg2 + k] * m.cheby[idx_seg2 * 3 + 2][0, k]
                            for k in range(N_seg2)
                        )
                    )

                m.control_derivative_knot_const = ConstraintList()
                for i in range(m.n_seg - 1):
                    for j in m.nu:
                        # add knot constraints for control derivative
                        m.control_derivative_knot_const.add(
                            control_derivative_knot_constraint(m, i, j)
                        )

    def ps_build(
        self,
        drone,
        obs,
        n_col,
        init=None,
        use_obstacles=True,
        psm_approx="chebyshev",
        knot_control_derivative=True,
    ):
        """Build model with Multi-Segment PseudoSpectral Optimal Control Problem with multirotor drone."""
        m = ConcreteModel()  # initialization of problem in form of Pyomo model
        # model and size of variables
        m.nx = RangeSet(0, drone.nx - 1)
        m.nu = RangeSet(0, drone.nu - 1)
        # number of collocation points in segments
        m.n_col = n_col
        # number of segments of multi-segment collocation
        m.n_seg = m.n_col.shape[0]
        m.n_col_sum = np.sum(m.n_col)  # total count of collocation points
        # progressive sum of collocation point (1st points of segments)
        m.n_col_add = np.zeros(m.n_col.shape, dtype=int)
        for i, cols in enumerate(m.n_col[:-1]):
            m.n_col_add[i + 1] = m.n_col_add[i] + cols

        m.cols = RangeSet(0, m.n_col_sum - 1)
        m.cols_w_bound = RangeSet(1, m.n_col_sum - 2)
        m.seg = RangeSet(0, m.n_seg - 1)
        m.knots = RangeSet(0, m.n_seg)
        m.nx_w_quat = [
            state for state in m.nx if state not in [6, 7, 8, 9]
        ]  # Range without quaternions
        m.knot_control_derivative = knot_control_derivative

        # state description
        # 0: rx, 1: ry, 2: rz, 3: vx, 4: vy, 5: vz, 6: qw, 7: qx, 8: qy, 9: qz, 10: wx, 11: wy, 12: wz
        m.x = Var(m.nx, m.cols, domain=Reals)
        # control description
        # 0: T, 1: taux, 2: tauy, 3: tauz
        m.u = Var(m.nu, m.cols, domain=Reals)
        m.t = Var(m.cols, domain=NonNegativeReals)

        # boundary condition for state
        for i in m.nx:
            m.x[i, 0].fix(drone.x0[i])
            m.x[i, m.n_col_sum - 1].fix(drone.xf[i])

        # boundary condition for control
        for i in m.nu:
            m.u[i, 0].fix(drone.uf[i])
            m.u[i, m.n_col_sum - 1].fix(drone.uf[i])

        # trajectory condition for state
        m.limit_state = ConstraintList()
        for i in m.nx:
            for j in m.cols:
                m.limit_state.add(drone.x_min[i] <= m.x[i, j])
                m.limit_state.add(m.x[i, j] <= drone.x_max[i])

        # trajectory condition for control
        m.limit_control = ConstraintList()
        for i in m.nu:
            for j in m.cols:
                m.limit_control.add(drone.u_min[i] <= m.u[i, j])
                m.limit_control.add(m.u[i, j] <= drone.u_max[i])

        # load or generate initial guess
        if init is None:
            x_guess = np.linspace(drone.x0, drone.xf, int(m.n_col_sum)).T
            u_guess = np.zeros((drone.nu, int(m.n_col_sum)))
            u_guess[0, :] = drone.mass * drone.g
            t_guess = np.linspace(
                drone.t0, (drone.tf_max - drone.tf_min) / 2, int(m.n_seg + 1)
            )
        else:
            x_guess = np.hstack([np.array(guess) for guess in init.state_guess])
            u_guess = np.hstack([np.array(guess) for guess in init.control_guess])
            t_guess = init.t_seg_guess
            # t_elem_guess = np.hstack([np.array(guess)
            #                     for guess in init.time_guess])

        # initialize variables with guess
        for i in m.cols:
            for j in m.nx:
                m.x[j, i] = x_guess[j, i]
        for i in m.cols:
            for j in m.nu:
                m.u[j, i] = u_guess[j, i]
        # for i in m.cols:
        #     m.t[i] = t_elem_guess[i]

        # create list with [t1,w1,D1,t2,w2,D2,...tN,wN,DN]
        # are gauss-lobatto grid points, integral weights and differential matrices
        # for 1...N segments
        # accessed with cheby[(i-1)*3+j], where i is segment and j is 0 for grid, 1 for weights and 2 for matrix
        m.cheby = []
        m.psm_approx = psm_approx
        if m.psm_approx == "chebyshev":
            from chebyshev import cheb

            for col in m.n_col:
                m.cheby.extend(cheb(col - 1))
        elif m.psm_approx == "legendre":
            from legendre import legendre

            for col in m.n_col:
                m.cheby.extend(legendre(col - 1))

        m.knot_time = Var(
            m.knots, domain=NonNegativeReals, bounds=(drone.t0, drone.tf_max)
        )

        # end time limit
        m.end_time_constraints = ConstraintList()
        m.end_time_constraints.add(m.t[m.n_col_sum - 1] <= drone.tf_max)
        m.end_time_constraints.add(m.t[m.n_col_sum - 1] >= drone.tf_min)

        # m.knot_time[m.n_seg] = 0.5*drone.tf_max

        # time has positive direction!
        m.knot_constraints = ConstraintList()
        for i in m.knots:
            if i < m.n_seg:
                m.knot_constraints.add(m.knot_time[i + 1] >= m.knot_time[i])

        # time has positive direction!
        m.time_constraints = ConstraintList()
        seg_start = 0
        for i in m.seg:
            if i > 0:
                seg_start = m.n_col_add[i]
            for j in range(1, m.n_col[i] - 2):
                m.time_constraints.add(m.t[j + 1 + seg_start] >= m.t[j + seg_start])

        # connect segments
        if m.n_seg > 1:
            m.limit_segments = ConstraintList()
            for i in m.n_col_add[1:]:
                for j in m.nx:
                    m.limit_segments.add(m.x[j, i - 1] == m.x[j, i])  # connect state
                for j in m.nu:
                    m.limit_segments.add(m.u[j, i - 1] == m.u[j, i])  # connect control
                m.limit_segments.add(m.t[i - 1] == m.t[i])  # connect time

        # time constraints
        m.time_definition = ConstraintList()
        for seg, n in enumerate(m.n_col):
            for i in range(n):
                m.time_definition.add(
                    m.t[i + m.n_col_add[seg]]
                    == (
                        (m.knot_time[seg + 1] - m.knot_time[seg])
                        * m.cheby[seg * 3 + 0][i]
                        + (m.knot_time[seg + 1] + m.knot_time[seg])
                    )
                    / 2
                )

        # m.initial_time_contraint = Constraint(
        #     expr=m.initial_time[0] == 0.0)
        m.knot_time[0].fix(drone.t0)
        for i in range(1, m.n_seg + 1):
            m.knot_time[i] = t_guess[i]

        # constraining obstacles
        if use_obstacles:
            m.limit_obstacles = ConstraintList()
            for i in m.cols:
                if obs.dim == 3:
                    for j in range(obs.map3d.shape[0]):
                        m.limit_obstacles.add(
                            (
                                (m.x[0, i] - obs.map3d[j, 0]) ** 2
                                + (m.x[1, i] - obs.map3d[j, 1]) ** 2
                                + (m.x[2, i] - obs.map3d[j, 2]) ** 2
                            )
                            >= (obs.space * np.sqrt(3) / 2 + drone.safe_radius) ** 2
                        )
                else:
                    for j in range(obs.map2d.shape[0]):
                        m.limit_obstacles.add(
                            (
                                (m.x[0, i] - obs.map2d[j, 0]) ** 2
                                + (m.x[1, i] - obs.map2d[j, 1]) ** 2
                            )
                            >= (obs.space * np.sqrt(2) / 2 + drone.safe_radius) ** 2
                        )

        self.ps_build_dynamics(m, drone)

        # # minimum time
        # m.obj = Objective(expr=m.t[m.n_col_sum-1], sense=minimize)
        desired_quaternion = drone.xf[6:10]
        # minimum state and control
        m.total_cost = 0
        for seg, n in enumerate(m.n_col):
            seg_start = m.n_col_add[seg]
            for i in range(n):
                m.total_cost = m.total_cost + (
                    (m.knot_time[seg + 1] - m.knot_time[seg]) * 0.5
                ) * m.cheby[seg * 3 + 1][i] * (
                    sum(
                        float(drone.Q[state, state])
                        * (m.x[state, seg_start + i] - drone.xf[state]) ** 2
                        for state in m.nx_w_quat
                    )
                    + sum(
                        float(drone.R[ctrl, ctrl])
                        * (m.u[ctrl, seg_start + i] - drone.uf[ctrl]) ** 2
                        for ctrl in m.nu
                    )
                    + float(
                        drone.Q[7, 7]
                        * quat_distance(
                            np.array(
                                [
                                    m.x[6, seg_start + i].value,
                                    m.x[7, seg_start + i].value,
                                    m.x[8, seg_start + i].value,
                                    m.x[9, seg_start + i].value,
                                ]
                            ),
                            desired_quaternion,
                        )
                    )
                )
        m.obj = Objective(expr=m.total_cost, sense=minimize)
        # m.obj = Objective(expr=1e-5*m.total_cost, sense=minimize)

        # # minimum control
        # m.total_cost = 0
        # for seg, n in enumerate(m.n_col):
        #     seg_start = m.n_col_add[seg]
        #     for i in range(n):
        #         m.total_cost = m.total_cost + ((m.knot_time[seg+1] - m.knot_time[seg]) * 0.5) * m.cheby[seg*3+1][i] * (sum(
        #             # state cost
        #             float(drone.R[ctrl, ctrl])*m.u[ctrl, seg_start + i]**2 for ctrl in m.nu))    # control cost
        # m.obj = Objective(expr=m.total_cost, sense=minimize)
        TransformationFactory("contrib.constraints_to_var_bounds").apply_to(m)
        # TransformationFactory('contrib.induced_linearity').apply_to(m)
        # TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
        # TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # TransformationFactory('contrib.aggregate_vars').apply_to(m)
        return m

    def ps_scale(self, m, drone):
        """Scale the model according to box constraints
        !!!Not working!!!
        """

        # scaling
        m.scaling_factor = Suffix(direction=Suffix.EXPORT)
        m.scaling_factor[m.obj] = 1e-5

        for col in m.cols:
            for state in m.nx_w_quat:
                m.scaling_factor[m.x[state, col]] = drone.x_max[state]
            for control in m.nu:
                m.scaling_factor[m.u[control, col]] = drone.u_max[control]

        # scaled_m = ScaleModel(create_new_model=True).create_using(m)
        scaled_m = TransformationFactory("core.scale_model").create_using(m)
        return scaled_m

    def ps_solve(
        self,
        model,
        scaled_model=None,
        max_iter=100,
        mu_init=1e-1,
        tol=1e-8,
        warm_push=0.001,
        warm_mult_push=0.001,
        log_path="no_path_log",
        warmstart=False,
    ):
        """Solve PYOMO problem with IPOPT."""
        if scaled_model is not None:
            m = scaled_model
        else:
            m = model
        # Create the ipopt solver plugin using the ASL interface
        solver = "ipopt"
        solver_io = "nl"
        stream_solver = True  # True prints solver output to screen
        keepfiles = True  # True prints intermediate file names (.nl,.sol,...)
        opt = SolverFactory(solver, solver_io=solver_io)
        # opt.reset()
        if opt is None:
            print("")
            print(
                "ERROR: Unable to create solver plugin for %s "
                "using the %s interface" % (solver, solver_io)
            )
            print("")
            exit(1)
        # Declare all suffixes
        # Ipopt bound multipliers (obtained from solution)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)

        # Ipopt bound multipliers (sent to solver)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        # Obtain dual solutions from first solve and send to warm start
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.presolve_time = 0
        m.solve_time = 0
        # Generate the constraint expression trees if necessary
        if solver_io != "nl":
            # only required when not using the ASL interface
            m.preprocess()
        if warmstart:
            # Send the model to ipopt and collect the solution
            print("")
            print("INITIAL SOLVE")
            # Set the max number of iterations
            opt.options["max_iter"] = int(max_iter)
            opt.options["tol"] = tol
            opt.options["mu_init"] = mu_init
            opt.options["halt_on_ampl_error"] = "yes"
            opt.options["nlp_scaling_method"] = "gradient-based"
            log_file = str(log_path + "_warmstart.log")
            results = opt.solve(
                m, keepfiles=keepfiles, tee=stream_solver, logfile=log_file
            )
            # load the results (including any values for previously declared
            # IMPORT / IMPORT_EXPORT Suffix components)
            m.solutions.load_from(results)
            # m.load(results)

            # Set Ipopt options for warm-start
            # The current values on the ipopt_zU_out and ipopt_zL_out suffixes will be used as initial
            # conditions for the bound multipliers to solve the new problem
            m.ipopt_zL_in.update(m.ipopt_zL_out)
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            opt.options["warm_start_init_point"] = "yes"
            opt.options["halt_on_ampl_error"] = "yes"

            opt.options["warm_start_bound_push"] = warm_push
            opt.options["warm_start_mult_bound_push"] = warm_mult_push
            m.presolve_time = opt._last_solve_time
        opt.options["tol"] = tol
        opt.options["mu_init"] = mu_init
        opt.options["nlp_scaling_method"] = "gradient-based"

        # #######################################################################################
        # # Send the model and suffix information to ipopt and collect the solution
        # print("")
        # print("WARM-STARTED SOLVE")
        # Set the max number of iterations
        opt.options["max_iter"] = int(
            max_iter
        )  # The solver plugin will scan the m for all active suffixes
        # valid for importing, which it will store into the results object
        if warmstart:
            log_file = str(log_path + "_after_warmstart.log")
        else:
            log_file = str(log_path + ".log")
        results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver, logfile=log_file)

        # load the results (including any values for previously declared
        # IMPORT / IMPORT_EXPORT Suffix components)
        m.solutions.load_from(results)
        if scaled_model is not None:
            ScaleModel.propagate_solution(
                scaled_model, scaled_model=m, original_model=model
            )
        m.solve_time = opt._last_solve_time


def example(save_results=False, file_name="example"):
    drone = Drone()
    sln = PsSolution()
    # generating the world
    folder = "traj_plan/worlds"
    trees = [
        Tree("tree_crown_1.txt", folder, 2),
        Tree("tree_crown_2.txt", folder, 3),
        Tree("tree_crown_4.txt", folder, 1),
    ]
    # obs = Gridmap('world_walls.txt', folder, dimension=2)
    world = Gridmap("world_orchard.txt", folder, trees=trees, dimension=3)
    # world limits, start and goal
    drone.x_min[0] = world.x_min
    drone.x_max[0] = world.x_max
    drone.x_min[1] = world.y_min
    drone.x_max[1] = world.y_max
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
    drone.tf_min = 0.0
    drone.tf_max = 300.0
    psc = PsControl()
    # number of collocation points in segments
    n_col = np.array([10], dtype=int)
    model = psc.ps_build(drone, world, n_col)
    psc.ps_solve(model=model, max_iter=300)

    if save_results:
        with open("traj_plan/data_traj/ps_" + file_name + ".pkl", mode="wb") as file:
            cloudpickle.dump(model, file)

    sln.new_solution(model, drone, world, fit="poly", fix_solution=False)
    sln.plot_sampled_with_col()


if __name__ == "__main__":
    example()
