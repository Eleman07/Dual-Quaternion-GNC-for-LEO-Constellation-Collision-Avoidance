"""
guidance.py
===========
Guidance layer for dual-quaternion 6-DOF LEO constellation collision avoidance.
Project #011 — Dual-Quaternion IEKF+LTV-LQR for LEO Constellation Collision Avoidance.

This module is responsible for:
    1. Generating the nominal reference trajectory (Clohessy-Wiltshire closed-form
       solution) that the LTV-LQR controller will track.
    2. Producing the time-varying linearization matrices A(t), B(t) of the relative
       translational dynamics about the nominal trajectory, consumed by the future
       control.py LTV-LQR module.
    3. Managing the keep-out exclusion volume around the target, including the
       interface through which the deep-learning perception module (LIDAR + optical
       fusion, target 3D shape estimation) updates the exclusion geometry online.

Architectural note
-------------------
guidance.py sits upstream in the GNC pipeline: dynamics.py (propagation) and the
future navigation.py (IEKF) / control.py (LTV-LQR) modules import and consume this
module — guidance.py does not import from them. Shared physical constants are
imported directly from dynamics.py, since they represent fixed environment/vehicle
configuration data rather than propagation logic.

References
----------
- Clohessy, W. H. & Wiltshire, R. S. (1960) "Terminal Guidance System for Satellite
  Rendezvous", Journal of the Aerospace Sciences.
- Filipe & Tsiotras (2014) "Adaptive Position and Attitude-Tracking Controller for
  Satellite Proximity Operations Using Dual Quaternions", JGCD.
- Fehse, W. (2003) "Automated Rendezvous and Docking of Spacecraft", Cambridge.
"""

import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Shared physical / mission constants (imported from dynamics.py)
# ---------------------------------------------------------------------------
from dynamics import (
    MU_EARTH,
    R_EARTH,
    R_REF,
    N_REF,
    V_REF,
    T_ORB,
    SAFE_RADIUS,
    APPROACH_VEL_MAX,
)


# ===========================================================================
# CATEGORY: Nominal trajectory generation (Clohessy-Wiltshire closed form)
# ===========================================================================

def cw_state_transition(t: float, n: float = N_REF) -> np.ndarray:
    """
    Clohessy-Wiltshire state transition matrix Phi(t) mapping an initial
    relative state [x, y, z, vx, vy, vz] (LVLH frame) to its value at time t,
    under the linearized (unperturbed) two-body relative dynamics.

    Convention: x = radial, y = along-track, z = cross-track.

    Parameters
    ----------
    t : float
        Propagation time [s].
    n : float
        Chief mean motion [rad/s] (default: reference shell mean motion).

    Returns
    -------
    Phi : (6, 6) ndarray
        CW state transition matrix such that state(t) = Phi(t) @ state(0).
    """
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    Phi_rr = np.array([
        [4 - 3 * c, 0.0, 0.0],
        [6 * (s - nt), 1.0, 0.0],
        [0.0, 0.0, c],
    ])
    Phi_rv = np.array([
        [s / n, 2 * (1 - c) / n, 0.0],
        [2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0.0],
        [0.0, 0.0, s / n],
    ])
    Phi_vr = np.array([
        [3 * n * s, 0.0, 0.0],
        [6 * n * (c - 1), 0.0, 0.0],
        [0.0, 0.0, -n * s],
    ])
    Phi_vv = np.array([
        [c, 2 * s, 0.0],
        [-2 * s, 4 * c - 3, 0.0],
        [0.0, 0.0, c],
    ])

    Phi = np.block([
        [Phi_rr, Phi_rv],
        [Phi_vr, Phi_vv],
    ])
    return Phi


def cw_propagate_state(state0: np.ndarray, t: float, n: float = N_REF) -> np.ndarray:
    """
    Propagate a CW relative state [x, y, z, vx, vy, vz] forward by time t using
    the closed-form analytic solution (no integration error, used as nominal
    reference / sanity check against the nonlinear propagator in dynamics.py).

    Parameters
    ----------
    state0 : (6,) ndarray
        Initial relative state [x, y, z, vx, vy, vz] in LVLH frame.
    t      : float
        Propagation time [s].
    n      : float
        Chief mean motion [rad/s].

    Returns
    -------
    state_t : (6,) ndarray
        Relative state at time t.
    """
    Phi = cw_state_transition(t, n)
    return Phi @ np.asarray(state0, dtype=np.float64)


def cw_glideslope_velocity(rho_vec: np.ndarray, t_go: float) -> np.ndarray:
    """
    Compute the required LVLH velocity for a straight-line, constant-time-of-flight
    glideslope approach: a simple guidance law that commands the chaser to close
    the relative position vector rho_vec to zero in exactly t_go seconds, moving
    along a straight line in the LVLH frame.

    This is used as a simple nominal guidance law for the final approach corridor,
    independent from (and typically more conservative than) the full CW two-impulse
    solution, intentionally limiting closing speed for collision-avoidance safety.

    Parameters
    ----------
    rho_vec : (3,) ndarray
        Current relative position [m] in LVLH frame (chaser w.r.t. target).
    t_go    : float
        Desired time-to-go [s] before reaching rho = 0.

    Returns
    -------
    v_cmd : (3,) ndarray
        Commanded LVLH velocity [m/s], clipped to the approach corridor speed limit.
    """
    if t_go <= 0.0:
        raise ValueError("t_go must be strictly positive.")

    v_cmd = -np.asarray(rho_vec, dtype=np.float64) / t_go
    speed = np.linalg.norm(v_cmd)
    if speed > APPROACH_VEL_MAX:
        v_cmd = v_cmd * (APPROACH_VEL_MAX / speed)
    return v_cmd


def generate_nominal_trajectory(state0: np.ndarray, t_span: float,
                                dt: float, n: float = N_REF):
    """
    Generate a discretized nominal (reference) trajectory using the CW closed-form
    solution, sampled at fixed time steps. This trajectory is the one the LTV-LQR
    controller will linearize about and track.

    Parameters
    ----------
    state0 : (6,) ndarray
        Initial relative state [x, y, z, vx, vy, vz] in LVLH frame.
    t_span : float
        Total trajectory duration [s].
    dt     : float
        Sampling interval [s].
    n      : float
        Chief mean motion [rad/s].

    Returns
    -------
    t_grid     : (N,) ndarray   Time samples [s].
    state_grid : (N, 6) ndarray Nominal state at each time sample.
    """
    t_grid = np.arange(0.0, t_span + dt, dt)
    state_grid = np.zeros((len(t_grid), 6), dtype=np.float64)

    for i,t in enumerate(t_grid):
        state_grid[i] = cw_propagate_state(state0, t, n)

    return t_grid, state_grid

