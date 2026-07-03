"""
control.py
==========
Control layer: LTV-LQR (Linear Time-Varying Linear Quadratic Regulator) for
dual-quaternion 6-DOF LEO constellation collision avoidance.
Project #011 — Dual-Quaternion IEKF+LTV-LQR for LEO Constellation Collision Avoidance.

This module is responsible for:
    1. Deriving the continuous-time linearized translational relative dynamics
       A(t), B(t) about the CW nominal trajectory (guidance.py only exposes
       the discrete-time CW state transition matrix Phi(t); this module
       supplies the continuous-time generator matrix A(t) it is the integral
       of, which the Riccati equation actually needs).
    2. Solving the time-varying LQR problem via backward-in-time integration
       of the differential Riccati equation over a receding horizon, and
       extracting the resulting time-varying feedback gain K(t).
    3. Producing the closed-loop control force command
           u(t) = -K(t) @ (x_hat(t) - x_nom(t))
       from the navigation.py state estimate and the guidance.py nominal
       trajectory, with an added soft keep-out barrier term so the commanded
       force also reacts to proximity with the exclusion volume, not only to
       tracking error (a pure LQR has no notion of a keep-out zone; the
       barrier term is what connects this controller to guidance.py's
       collision-avoidance geometry).
    4. Producing an independent PD-type attitude control torque about the
       nominal (identity, in this baseline) attitude, since the CW/LQR
       translational solution above says nothing about attitude and
       dynamics.py's dynamics_rhs consumes a torque input separately from
       the LQR force.

Architectural note
-------------------
control.py sits downstream of BOTH dynamics.py (physical constants: MASS,
INERTIA, THRUST_MAX -- needed to build B(t) and to saturate commands) and
guidance.py (nominal trajectory x_nom(t), keep-out geometry SAFE_RADIUS).
Neither dynamics.py nor guidance.py import from control.py, preserving the
project's upstream/downstream layering. control.py receives the estimated
state from navigation.py as a plain (14,) array at call time (via
NavigationState.x from the caller, e.g. plant_simulator.py's compute_control
hook) -- it does NOT import navigation.py, since the estimator and the
controller are independent GNC blocks that only need to agree on the state
layout, not on each other's implementation.

Why LQR instead of a learned (DNN) policy
-------------------------------------------
Collision avoidance is the safety-critical decision step of this pipeline.
LQR gives a provably stabilizing, optimal-w.r.t.-a-declared-cost feedback law
for the (locally) linear CW dynamics, with a closed-form Riccati solution
that can be inspected and verified. A learned policy would not carry those
guarantees. Deep learning in this project is confined to perception (target
3D shape estimation from LIDAR+optical fusion, feeding guidance.py's
exclusion volume) rather than to the avoidance decision itself.

State / control convention
----------------------------
Full navigation state (14 components), same layout as dynamics.py / navigation.py:
    x[0:8]  = dq  (dual quaternion, [q_r | q_d])
    x[8:14] = dv  (twist, [omega | v_rel])

LQR operates on the reduced 6-component TRANSLATIONAL state only,
[x, y, z, vx, vy, vz] (LVLH frame) -- the same layout guidance.py's CW
functions use -- extracted from the full 14-component state via dq_to_pose /
dv slicing. Attitude control is handled separately (category 4 above) since
CW is a translational-only model.

References
----------
- Lewis, Vrabie & Syrmos (2012) "Optimal Control", 3rd ed., Wiley.
  (Differential Riccati equation, time-varying LQR)
- Clohessy, W. H. & Wiltshire, R. S. (1960), as cited in guidance.py.
- Ames, Coogan, Egerstedt et al. (2019) "Control Barrier Functions: Theory
  and Applications", ECC. (Soft keep-out barrier term)
- Filipe & Tsiotras (2014), as cited throughout dynamics.py / navigation.py.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from dataclasses import dataclass, field

from dynamics import (
    MASS,
    INERTIA,
    INERTIA_INV,
    THRUST_MAX,
    N_REF,
    dq_to_pose,
)
from guidance import (
    cw_propagate_state,
    cw_state_transition,
    generate_nominal_trajectory,
    SAFE_RADIUS,
)

POS_DIM = 3
TRANS_DIM = 6  # [x, y, z, vx, vy, vz]


# ===========================================================================
# CATEGORY: Continuous-time CW linearization  A(t), B(t)
# ===========================================================================
#
# guidance.py exposes the CLOSED-FORM DISCRETE transition Phi(t) (state(t) =
# Phi(t) @ state(0)), which is exactly right for propagating a nominal
# trajectory but is not what a Riccati equation needs: the differential
# Riccati equation is driven by the continuous-time GENERATOR matrix A(t)
# such that xdot = A(t) x + B(t) u, i.e. the CW equations of motion
# themselves, not their integral. Unlike the general LTV case, CW's A is
# actually time-invariant (the relative dynamics about a circular reference
# orbit do not depend on the true anomaly), so "A(t)" here is really a
# constant A -- kept as a function of a mean-motion argument for interface
# generality (e.g. a future eccentric-reference extension), and because the
# "LTV" in LTV-LQR still holds through the *nominal trajectory* x_nom(t) the
# controller tracks, even though A itself is constant for this CW-linearized
# case.
# ===========================================================================

def cw_A_matrix(n: float = N_REF) -> np.ndarray:
    """
    Continuous-time CW state matrix A such that xdot = A @ x for the
    unforced translational relative dynamics, x = [x,y,z,vx,vy,vz] (LVLH).

    Derivation: the classical CW / Hill's equations,
        xddot = 3 n^2 x + 2 n ydot
        yddot = -2 n xdot
        zddot = -n^2 z
    written in first-order form. This is the continuous-time generator whose
    matrix exponential exp(A t) reproduces guidance.cw_state_transition(t)
    exactly (verified numerically in cw_A_matrix_check).

    Parameters
    ----------
    n : float  Chief mean motion [rad/s] (default: reference shell mean motion).

    Returns
    -------
    A : (6, 6) ndarray
    """
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0] = 3.0 * n ** 2
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    A[5, 2] = -n ** 2
    return A


def cw_A_matrix_check(n: float = N_REF, t_test: float = 300.0,
                      tol: float = 1e-6) -> dict:
    """
    Numerically verify cw_A_matrix by comparing exp(A * t_test) against
    guidance.cw_state_transition(t_test) -- the two must agree, since Phi(t)
    is defined as the state transition matrix (the matrix exponential of the
    generator A) of the same physical system.

    Parameters
    ----------
    n      : float  Chief mean motion [rad/s].
    t_test : float  Propagation time [s] used for the check.
    tol    : float  Max allowed entrywise absolute difference.

    Returns
    -------
    report : dict with 'max_abs_diff' and 'passed'.
    """
    A = cw_A_matrix(n)
    Phi_from_A = expm(A * t_test)
    Phi_closed_form = cw_state_transition(t_test, n)
    diff = np.abs(Phi_from_A - Phi_closed_form)
    return {"max_abs_diff": float(np.max(diff)), "passed": bool(np.max(diff) < tol)}


def cw_B_matrix(mass: float = MASS) -> np.ndarray:
    """
    Continuous-time CW input matrix B such that xdot = A @ x + B @ u, where
    u is a translational control FORCE [N] in the LVLH frame (force, not
    acceleration -- consistent with dynamics_rhs's u_force argument).

    Returns
    -------
    B : (6, 3) ndarray
    """
    B = np.zeros((6, 3))
    B[3:6, :] = np.eye(3) / mass
    return B


# ===========================================================================
# CATEGORY: Time-varying LQR — differential Riccati equation
# ===========================================================================

def _riccati_rhs(tau: float, P_flat: np.ndarray, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray, R_inv: np.ndarray) -> np.ndarray:
    """
    Right-hand side of the differential Riccati equation, integrated forward
    in the reversed-time variable tau (see solve_lqr_gain_schedule for the
    tau = t_span - t substitution):
        dP/dtau = A^T P + P A - P B R^-1 B^T P + Q

    Parameters
    ----------
    tau    : float          Reversed-time variable (unused explicitly here
                             since A, B are time-invariant for the CW case --
                             see cw_A_matrix docstring -- but kept in the
                             signature for solve_ivp and for interface
                             generality with a future eccentric or
                             J2-augmented A(t)).
    P_flat : (36,) ndarray  Flattened symmetric P.
    A, B   : ndarrays        System matrices.
    Q      : (6,6) ndarray   State cost.
    R_inv  : (3,3) ndarray   Inverse of control cost R.

    Returns
    -------
    Pdot_flat : (36,) ndarray  Flattened dP/dtau.
    """
    P = P_flat.reshape(TRANS_DIM, TRANS_DIM)
    Pdot = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    return Pdot.flatten()


def solve_lqr_gain_schedule(t_span: float, dt: float, Q: np.ndarray,
                            R: np.ndarray, P_terminal: np.ndarray = None,
                            n: float = N_REF, mass: float = MASS):
    """
    Solve the finite-horizon differential Riccati equation backward in time
    over [0, t_span] and return the resulting time-varying gain schedule
    K(t) = R^-1 B^T P(t), sampled at the same dt grid guidance.py's
    generate_nominal_trajectory uses, so the two can be zipped together.

    Backward integration: standard LQR theory poses the Riccati ODE with a
    terminal condition P(t_span) = P_terminal and integrates toward t=0. This
    is implemented here by substituting tau = t_span - t (so tau runs
    forward from 0, at t=t_span, to t_span, at t=0), integrating forward in
    tau with solve_ivp, then reversing the result back onto the original
    time grid -- avoids relying on solve_ivp's negative-dt integration
    semantics.

    Parameters
    ----------
    t_span      : float        Horizon duration [s].
    dt          : float        Sampling interval [s] for the returned schedule.
    Q           : (6,6) ndarray State cost (weights position/velocity tracking
                                 error -- larger entries penalize error more).
    R           : (3,3) ndarray Control cost (weights force effort/fuel use).
    P_terminal  : (6,6) ndarray Terminal Riccati condition (default: Q, i.e.
                                 the standard "no extra terminal penalty
                                 beyond the running cost" choice).
    n           : float         Chief mean motion [rad/s].
    mass        : float         Chaser mass [kg], for B.

    Returns
    -------
    t_grid   : (N,) ndarray      Time samples [s], 0 ... t_span.
    K_grid   : (N, 3, 6) ndarray Feedback gain K(t) at each time sample, such
                                 that u(t) = -K(t) @ (x(t) - x_nom(t)).
    """
    if P_terminal is None:
        P_terminal = Q.copy()

    A = cw_A_matrix(n)
    B = cw_B_matrix(mass)
    R_inv = np.linalg.inv(R)

    t_grid = np.arange(0.0, t_span + dt, dt)
    tau_eval = t_grid

    sol = solve_ivp(
        fun=lambda tau, P_flat : _riccati_rhs(tau, P_flat, A, B, Q, R_inv),
        t_span=(0.0, t_span),
        y0 = P_terminal.flatten(),
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    P_of_tau = sol.y.T.reshape(-1, TRANS_DIM, TRANS_DIM)
    P_of_t = P_of_tau[::-1]
    K_grid = np.zeros((len(t_grid), 3, TRANS_DIM))
    for k in range(len(t_grid)):
        K_grid[k] = R_inv @ B.T @ P_of_t[k]

    return t_grid, K_grid


def solve_lqr_gain_steady_state(Q: np.ndarray, R: np.ndarray,
                                n: float = N_REF, mass: float = MASS) -> np.ndarray:
    """
    Solve the steady-state (algebraic, infinite-horizon) Riccati equation
    instead of the finite-horizon differential one.

    Since A is time-invariant for the CW-linearized problem (see cw_A_matrix
    docstring), the "genuinely time-varying" finite-horizon Riccati solution
    from solve_lqr_gain_schedule converges to this steady-state gain as the
    horizon grows -- but only slowly relative to the orbital period T_ORB
    (empirically, a horizon of several orbital periods is needed for the
    early-time gain to approach steady state to within a few percent). This
    algebraic solve is the numerically preferred choice for a CW-tracking
    controller precisely because A, B are constant: it is exact, cheap (no
    ODE integration), and free of the horizon-length sensitivity that makes
    solve_lqr_gain_schedule's short-horizon gains marginally stable or
    unstable in closed loop. solve_lqr_gain_schedule is kept for genuinely
    time-varying extensions of A(t) (e.g. eccentric reference orbits) where
    no algebraic steady-state solution exists.

    Parameters
    ----------
    Q    : (6,6) ndarray  State cost.
    R    : (3,3) ndarray  Control cost.
    n    : float           Chief mean motion [rad/s].
    mass : float           Chaser mass [kg].

    Returns
    -------
    K : (3, 6) ndarray  Constant steady-state feedback gain,
        u(t) = -K @ (x(t) - x_nom(t)).
    """
    from scipy.linalg import solve_continuous_are
    A = cw_A_matrix(n)
    B = cw_B_matrix(mass)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def lqr_gain_at(t: float, t_grid: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    """
    Look up (nearest-neighbor) the feedback gain K at time t from a
    precomputed gain schedule, for use inside a real-time control loop where
    re-solving the Riccati equation every cycle would be wasteful.

    Parameters
    ----------
    t      : float               Query time [s].
    t_grid : (N,) ndarray         Time samples from solve_lqr_gain_schedule.
    K_grid : (N, 3, 6) ndarray    Gain schedule from solve_lqr_gain_schedule.

    Returns
    -------
    K : (3, 6) ndarray  Gain at (or nearest to) time t.
    """
    idx = int(np.clip(np.searchsorted(t_grid, t), 0, len(t_grid) - 1))
    return K_grid[idx]


# ===========================================================================
# CATEGORY: Keep-out soft barrier (connects control.py to guidance.py's
# exclusion geometry)
# ===========================================================================

def keepout_barrier_force(rel_pos: np.ndarray, safe_radius: float = SAFE_RADIUS,
                          gain: float = 1.0, activation_margin: float = 2.0) -> np.ndarray:
    """
    Soft repulsive force pushing the chaser away from the keep-out sphere
    around the target, active only within activation_margin * safe_radius so
    it does not perturb tracking far from the exclusion zone. This is what
    lets the LTV-LQR tracking law also behave as a collision-avoidance
    controller rather than a pure trajectory-follower: without this term the
    LQR would happily fly straight through the nominal trajectory even if
    that trajectory (or a deviation from it) intersected the keep-out volume.

    Barrier form: an inverse-distance repulsion, finite and continuous at the
    activation boundary (zero force exactly at
    ||rel_pos|| = activation_margin * safe_radius, growing without bound as
    ||rel_pos|| -> safe_radius), a standard, simple control-barrier-style
    penalty (see Ames et al. 2019, cited in the module docstring) chosen for
    its closed-form simplicity over a full constrained-QP CBF-LQR fusion,
    which is out of scope for this baseline.

    Parameters
    ----------
    rel_pos           : (3,) ndarray  Current relative position [m], LVLH.
    safe_radius        : float         Keep-out sphere radius [m].
    gain               : float         Barrier force scale [N * m].
    activation_margin  : float         Multiplier on safe_radius defining
                                        where the barrier begins to act.

    Returns
    -------
    f_barrier : (3,) ndarray  Repulsive force [N], zero outside the
        activation shell, directed away from the target (origin of rel_pos).
    """
    r = np.asarray(rel_pos, dtype=np.float64)
    dist = np.linalg.norm(r)
    activation_dist = activation_margin * safe_radius

    if dist >= activation_dist or dist < 1e-6:
        return np.zeros(3)

    direction = r / dist
    # Zero at dist = activation_dist, -> large as dist -> safe_radius.
    magnitude = gain * (
        1.0 / max(dist - safe_radius, 1e-3)
        - 1.0 / (activation_dist - safe_radius)
    )
    magnitude = max(magnitude, 0.0)
    return direction * magnitude


# ===========================================================================
# CATEGORY: Attitude control (independent PD loop)
# ===========================================================================
#
# CW/LQR above is a translational-only model; dynamics_rhs also consumes a
# torque input, so a minimal PD attitude controller is provided to drive the
# relative attitude toward identity (chaser aligned with the LVLH frame),
# independent of the translational LQR loop. A full attitude-inclusive
# LTV-LQR on the dual-quaternion kinematics is a natural extension (per
# Filipe & Tsiotras 2014, cited in the module docstring) but is out of scope
# for this baseline controller.
# ===========================================================================

def attitude_pd_torque(q_att: np.ndarray, omega: np.ndarray,
                       kp: float = 0.5, kd: float = 2.0) -> np.ndarray:
    """
    PD control torque driving the relative attitude quaternion toward
    identity [1,0,0,0] and angular velocity toward zero.

    Parameters
    ----------
    q_att : (4,) ndarray  Current relative attitude quaternion [w,x,y,z].
    omega : (3,) ndarray  Current relative angular velocity [rad/s], body frame.
    kp    : float          Proportional (attitude error) gain.
    kd    : float          Derivative (rate) gain.

    Returns
    -------
    u_torque : (3,) ndarray  Commanded torque [N*m], body frame.
    """
    q = np.asarray(q_att, dtype=np.float64)
    if q[0] < 0.0:
        q = -q  # shortest-path convention, same as navigation.pose_innovation
    # Small-angle attitude error vector (vector part of the error quaternion
    # relative to identity, which is just q's own vector part since the
    # target attitude here is identity).
    theta_err = 2.0 * q[1:4]
    return -kp * theta_err - kd * np.asarray(omega, dtype=np.float64)


# ===========================================================================
# CATEGORY: Top-level control law
# ===========================================================================

@dataclass
class ControllerConfig:
    """
    Configuration bundle for the LTV-LQR + keep-out barrier + attitude-PD
    controller.

    Attributes
    ----------
    Q                  : (6,6) ndarray  LQR state cost (position/velocity error).
    R                  : (3,3) ndarray  LQR control cost (force effort).
    horizon            : float          Riccati integration horizon [s].
    dt                 : float          Gain-schedule sampling interval [s].
    barrier_gain       : float          keepout_barrier_force gain.
    barrier_margin     : float          keepout_barrier_force activation_margin.
    att_kp, att_kd     : float          attitude_pd_torque gains.
    force_limit        : float          Per-axis force saturation [N] (thruster limit).
    """
    Q: np.ndarray = field(default_factory=lambda: np.diag(
        [1e-4, 1e-4, 1e-4, 1.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e6)
    use_steady_state_gain: bool = True
    # horizon/dt only matter when use_steady_state_gain=False. A CW-tracking
    # LQR needs several orbital periods of finite-horizon Riccati integration
    # before the early-time gain approaches the steady-state (algebraic)
    # solution -- see solve_lqr_gain_steady_state's docstring -- so the
    # default here is set to ~9 orbital periods (T_ORB ~= 5676 s) as a safety
    # margin, rather than the much shorter (and closed-loop unstable) horizon
    # a naive one-orbit choice would give.
    horizon: float = 50_000.0
    dt: float = 10.0
    barrier_gain: float = 50.0
    barrier_margin: float = 2.0
    att_kp: float = 0.5
    att_kd: float = 2.0
    force_limit: float = THRUST_MAX


class LTVLQRController:
    """
    Stateful controller wrapper: precomputes the LQR gain schedule once
    (expensive: one Riccati integration) against a given nominal trajectory,
    then offers a cheap per-cycle __call__ suitable for a real-time control
    loop such as plant_simulator.py's compute_control hook.

    Usage
    -----
        ctrl = LTVLQRController(config, state0_nominal)
        ...
        u_force, u_torque = ctrl(nav_state_estimate_14vec, t)
    """

    def __init__(self, config: ControllerConfig, nominal_state0: np.ndarray,
                 n: float = N_REF, mass: float = MASS):
        self.config = config
        self.n = n
        self.mass = mass

        self.t_nom, self.x_nom = generate_nominal_trajectory(
            nominal_state0, config.horizon, config.dt, n)

        if config.use_steady_state_gain:
            # Constant gain (see solve_lqr_gain_steady_state docstring for
            # why this is the numerically preferred default for the
            # time-invariant CW A matrix). Stored as a length-1 schedule so
            # lqr_gain_at's lookup logic works unchanged.
            K_const = solve_lqr_gain_steady_state(config.Q, config.R, n=n, mass=mass)
            self.t_gain = np.array([0.0])
            self.K_gain = K_const.reshape(1, 3, TRANS_DIM)
        else:
            self.t_gain, self.K_gain = solve_lqr_gain_schedule(
                config.horizon, config.dt, config.Q, config.R, n=n, mass=mass)

    def _nominal_state_at(self, t: float) -> np.ndarray:
        """Nearest-neighbor lookup of the precomputed nominal trajectory."""
        idx = int(np.clip(np.searchsorted(self.t_nom, t), 0, len(self.t_nom) - 1))
        return self.x_nom[idx]

    def __call__(self, x_hat: np.ndarray, t: float):
        """
        Compute the closed-loop control command for the current cycle.

        Parameters
        ----------
        x_hat : (14,) ndarray  Full navigation state estimate [dq | dv]
                                (e.g. NavigationState.x from navigation.py).
        t     : float           Elapsed mission time [s], used to look up the
                                 nominal trajectory and gain schedule.

        Returns
        -------
        u_force  : (3,) ndarray  Commanded translational force [N], LVLH frame,
                                  saturated to +/- config.force_limit per axis.
        u_torque : (3,) ndarray  Commanded attitude torque [N*m], body frame.
        """
        r_hat, q_att_hat = dq_to_pose(x_hat[0:8])
        omega_hat = x_hat[8:11]
        v_hat = x_hat[11:14]
        x_trans_hat = np.concatenate([r_hat, v_hat])

        x_trans_nom = self._nominal_state_at(t)
        K = lqr_gain_at(t, self.t_gain, self.K_gain)

        u_lqr = -K @ (x_trans_hat - x_trans_nom)
        u_barrier = keepout_barrier_force(
            r_hat, gain=self.config.barrier_gain,
            activation_margin=self.config.barrier_margin)

        u_force = np.clip(u_lqr + u_barrier,
                          -self.config.force_limit, self.config.force_limit)

        u_torque = attitude_pd_torque(
            q_att_hat, omega_hat, kp=self.config.att_kp, kd=self.config.att_kd)

        return u_force, u_torque

