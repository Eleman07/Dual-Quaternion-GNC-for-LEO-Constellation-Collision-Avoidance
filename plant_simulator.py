"""
plant_simulator.py
===================
Hardware-in-the-loop bridge between Kerbal Space Program 1 (via the kRPC mod)
and the Python GNC pipeline (dynamics.py / guidance.py / navigation.py).
Project #011 — Dual-Quaternion IEKF+LTV-LQR for LEO Constellation Collision Avoidance.

Role in the architecture
-------------------------
KSP is used as the "plant": the CHASER is a real vessel flying in KSP, and its
true relative state is read back through kRPC every control cycle. The TARGET
(the object to avoid) is NOT a second KSP vessel — it is a virtual state
propagated in Python with guidance.cw_propagate_state, offset from a reference
circular orbit consistent with dynamics.R_REF / dynamics.N_REF. This matches
project objective #5 (end-to-end simulation, real-time feasibility per cycle)
while keeping the collision-avoidance target abstract, as called for in the
brief (the target is "un detriti o un altro satellite" whose 3D shape is
supplied by the perception stack, not necessarily a flyable KSP vessel).

Per control cycle this module:
    1. Reads the chaser's true position/velocity/attitude/angular velocity
       from KSP via kRPC (SpaceCenter.Vessel), in the vessel's orbital
       reference frame (an LVLH-like frame: this is the same radial /
       along-track / normal convention dynamics.py's LVLH frame assumes,
       modulo an axis-order remap handled explicitly below).
    2. Propagates the virtual target state analytically for comparison.
    3. Builds the relative dual-quaternion state [dq | dv] using the exact
       dynamics.py convention (dq_from_pose, [w,x,y,z] quaternion order),
       converting kRPC's native (x,y,z,w) quaternion order explicitly.
    4. Feeds a (noise-corrupted, sensor-like) measurement of that relative
       pose into navigation.iekf_step to obtain the filtered estimate.
    5. Sends whatever control force/torque command it is given (produced by
       the future control.py LTV-LQR, or a placeholder here) back to KSP as
       RCS translation + reaction-wheel/RCS rotation inputs.

Architectural note
-------------------
plant_simulator.py sits at the very bottom of the stack: it imports from
dynamics.py, guidance.py and navigation.py, but none of those modules import
from it (same "upstream/downstream" rule already used throughout the
project). control.py, once written, is expected to plug into the
`compute_control` hook below rather than being imported top-down by it.

Coordinate frame caveat
------------------------
kRPC's Vessel.orbital_reference_frame is: x = out of the orbit (radial,
away from body center growing outward through the vessel), y = orbit
prograde (along-track), z = orbit normal. dynamics.py's LVLH convention
(see its module docstring and dynamics_rhs) is x = radial, y = along-track,
z = cross-track, i.e. the SAME axis semantics, so no axis permutation is
required going into dq_from_pose — only the quaternion component order
(kRPC: (x,y,z,w), dynamics.py: [w,x,y,z]) needs converting.

Dependencies
------------
    pip install krpc
KSP 1 must be running with the kRPC server mod started and a chaser vessel
in orbit near the reference shell altitude (ALT_REF in dynamics.py) as the
active vessel (or looked up by name — see PlantConfig.vessel_name).

References
----------
- kRPC documentation: https://krpc.github.io/krpc/
- Filipe & Tsiotras (2014), as cited throughout dynamics.py / navigation.py.
"""
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "gnc"))
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import krpc
    KRPC_AVAILABLE = True
except ImportError:
    # krpc (and a running KSP instance) is only required for the live
    # hardware-in-the-loop bridge (KSPPlant / run() below). The offline
    # closed-loop simulation and convergence plots further down in this
    # module (OfflineSimConfig / simulate_closed_loop / plot_convergence)
    # do not touch KSP at all, so they must stay importable/runnable even
    # without krpc installed or KSP running. KSPPlant.__init__ still fails
    # loudly (via krpc.connect) if someone actually tries to instantiate it
    # without krpc present.
    krpc = None
    KRPC_AVAILABLE = False

from dynamics import (
    MASS,
    R_REF,
    N_REF,
    ALT_REF,
    dq_from_pose,
    dq_to_pose,
    dq_mult,
    dq_conj,
    quat_mult,
    quat_conj,
)
from guidance import cw_propagate_state
from navigation import (
    NavigationState,
    iekf_step,
    measurement_model,
)
from control import LTVLQRController, ControllerConfig


# ===========================================================================
# CATEGORY: Configuration
# ===========================================================================

@dataclass
class PlantConfig:
    """
    Configuration for the KSP <-> GNC bridge.

    Attributes
    ----------
    vessel_name       : str or None
        Name of the chaser vessel in KSP to control. If None, uses whichever
        vessel is currently active in-game (space_center.active_vessel).
    dt                : float
        Control-loop period [s]. Also used as the IEKF propagation step and
        as the target's CW propagation step.
    target_offset0    : (6,) ndarray
        Initial relative state of the virtual target w.r.t. the chaser's
        reference orbit, [x,y,z,vx,vy,vz] in LVLH, at simulation start
        (t=0). The target is then propagated forward with
        guidance.cw_propagate_state; the chaser is whatever KSP reports.
    pos_noise_std     : float
        Standard deviation [m] of synthetic position measurement noise
        added before feeding the IEKF (stands in for the LIDAR+optical
        perception stack's sensor noise until that module exists).
    att_noise_std_rad : float
        Standard deviation [rad] of synthetic small-angle attitude
        measurement noise.
    max_cycles        : int or None
        Stop after this many control cycles (None = run until interrupted).
    """
    vessel_name : Optional[str] = None
    dt : float = 1.0
    target_offset0: np.ndarray = field(
        default_factory=lambda: np.array([100.0, -300.0, 20.0, 0.0, 0.0, 0.0]))
    pos_noise_std: float = 0.5
    att_noise_std_rad: float = 0.01
    max_cycles: Optional[int] = None


# ===========================================================================
# CATEGORY: kRPC <-> dynamics.py conventions
# ===========================================================================

def krpc_quat_to_dynamics(q_xyzw) -> np.ndarray:
    """
    Convert a kRPC quaternion tuple (x, y, z, w) to dynamics.py's [w, x, y, z]
    convention. Getting this wrong silently rotates every downstream attitude
    computation, so it is kept as an isolated, single-purpose function rather
    than inlined at each call site.

    Parameters
    ----------
    q_xyzw : tuple/(4,) array  kRPC-convention quaternion (x, y, z, w).

    Returns
    -------
    q_wxyz : (4,) ndarray  dynamics.py-convention quaternion [w, x, y, z].
    """
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def dynamics_quat_to_krpc(q_wxyz: np.ndarray):
    """Inverse of krpc_quat_to_dynamics: [w,x,y,z] -> kRPC (x,y,z,w) tuple."""
    w, x, y, z = q_wxyz
    return (float(x), float(y), float(z), float(w))


# ===========================================================================
# CATEGORY: KSP connection and telemetry read-back
# ===========================================================================

class KSPPlant:
    """
    Thin wrapper around a kRPC connection exposing exactly the read/write
    surface the GNC loop needs: read chaser pose+twist, write force/torque
    commands. Keeping this separate from the GNC math (which stays pure
    numpy, no kRPC objects) means dynamics.py / navigation.py never need to
    know kRPC exists, preserving the "downstream modules don't import
    upstream" rule used throughout the project.
    """

    def __init__(self, config: PlantConfig):
        if not KRPC_AVAILABLE:
            raise ImportError(
                "KSPPlant requires the 'krpc' package (pip install krpc) "
                "and a running KSP 1 instance with the kRPC server mod "
                "started. For an offline (no-KSP) closed-loop run, use "
                "simulate_closed_loop() / plot_convergence() instead."
            )
        self.config = config
        self.conn = krpc.connect(name="Project011-GNC-Bridge")
        self.sc = self.conn.space_center
        self.vessel = (
            self.sc.active_vessel if config.vessel_name is None
            else next(v for v in self.sc.vessels if v.name == config.vessel_name)
        )
        # LVLH-like frame: radial / prograde / normal, centered on the
        # vessel's own orbit — matches dynamics.py's [radial, along-track,
        # cross-track] LVLH axis semantics (see module docstring above).
        self.orbital_frame = self.vessel.orbital_reference_frame
        # RCS translation and pitch/yaw/roll are additive by default; make
        # that explicit so external SAS/keyboard input can't fight the loop.
        self.vessel.control.input_mode = self.sc.ControlInputMode.override
        self.vessel.control.rcs = True

    def read_chaser_dq_state(self) -> np.ndarray:
        """
        Read the chaser's current pose and twist from KSP and pack them into
        the same 14-component [dq | dv] flat state layout dynamics.py uses.

        NOTE: KSP reports the chaser's *absolute* orbital-frame state, not a
        relative-to-target state. dq_from_pose here encodes the chaser's pose
        directly; the caller (control loop below) is responsible for
        differencing this against the virtual target's state to get the
        actual relative state the navigation/guidance/control stack expects.

        Returns
        -------
        state : (14,) ndarray  [dq(8) | dv(6)] for the chaser alone.
        """
        pos = np.array(self.vessel.position(self.orbital_frame), dtype=np.float64)
        vel = np.array(self.vessel.velocity(self.orbital_frame), dtype=np.float64)
        q_krpc = self.vessel.rotation(self.orbital_frame)
        q_att = krpc_quat_to_dynamics(q_krpc)
        omega_krpc = np.array(
            self.vessel.angular_velocity(self.orbital_frame), dtype=np.float64)

        dq = dq_from_pose(pos, q_att)
        dv = np.concatenate([omega_krpc, vel])
        return np.concatenate([dq, dv])

    def send_control(self, u_force_body: np.ndarray, u_torque_body: np.ndarray):
        """
        Apply a commanded body-frame force/torque to the chaser using RCS
        translation and combined RCS+reaction-wheel rotation inputs.

        kRPC does not expose a direct Newton/Newton-metre command interface
        (KSP's control model is normalized [-1, 1] stick inputs, not raw
        force), so commands are scaled by the vessel's currently *available*
        max RCS force/torque, per axis, and clamped to [-1, 1]. This means
        the achieved force/torque is only as accurate as the vessel's control
        authority — same real-world caveat as any actuator saturation model.

        Parameters
        ----------
        u_force_body  : (3,) ndarray  Desired control force [N], vessel frame.
        u_torque_body : (3,) ndarray  Desired control torque [N*m], vessel frame.
        """
        ctrl = self.vessel.control
        max_force = np.array(self.vessel.available_rcs_force, dtype=np.float64)
        if max_force.ndim > 1:
            # available_rcs_force can come back as (2,3) [neg/pos triplets]
            # or (3,2) [per-axis +/- pairs] depending on kRPC/KSP version;
            # collapse whichever axis has length 2, keep the axis of length 3.
            axis_to_collapse = max_force.shape.index(2) if 2 in max_force.shape else -1
            max_force = np.max(np.abs(max_force), axis=axis_to_collapse)
        max_force = np.where(np.abs(max_force) < 1e-6, 1.0, np.abs(max_force))
        f_cmd = np.clip(u_force_body / max_force, -1.0, 1.0).flatten()
        # kRPC control axes: right(x) / forward(y) / bottom(z) -- forward maps
        # to the .forward input, right to .right, "up" is -bottom.
        ctrl.right = float(f_cmd[0])
        ctrl.forward = float(f_cmd[1])
        ctrl.up = float(-f_cmd[2])

        max_torque = np.array(self.vessel.available_torque[0], dtype=np.float64)
        if max_torque.ndim > 1:
            axis_to_collapse = max_torque.shape.index(2) if 2 in max_torque.shape else -1
            max_torque = np.max(np.abs(max_torque), axis=axis_to_collapse)
        max_torque = np.where(np.abs(max_torque) < 1e-6, 1.0, np.abs(max_torque))
        t_cmd = np.clip(u_torque_body / max_torque, -1.0, 1.0).flatten()

        ctrl.pitch = float(t_cmd[0])
        ctrl.roll = float(t_cmd[1])
        ctrl.yaw = float(t_cmd[2])

    def universal_time(self) -> float:
        """Current KSP universal time [s], for logging/propagation timing."""
        return self.sc.ut


# ===========================================================================
# CATEGORY: Relative-state assembly (chaser telemetry - virtual target)
# ===========================================================================

def relative_state_from_absolute(chaser_state: np.ndarray,
                                 target_state_lvlh: np.ndarray) -> np.ndarray:
    """
    Build the CHASER-RELATIVE-TO-TARGET dual-quaternion state that
    navigation.py / dynamics.py expect, from the chaser's absolute
    orbital-frame pose (read from KSP) and the virtual target's LVLH offset
    state (propagated analytically with guidance.cw_propagate_state).

    Convention: the target is defined as the chaser's *reference* orbit
    offset by target_offset0 and propagated with the CW closed-form
    solution (guidance.cw_propagate_state), i.e. the "chief" of the CW
    problem is the chaser's nominal circular orbit and the target is the
    "deputy" — the mirror image of how dynamics.py treats the chaser as
    deputy relative to a target chief. This is an explicit modeling choice
    (documented here) rather than a hidden assumption: since only the
    relative geometry matters for collision avoidance, either body may be
    designated chief without loss of generality, and this bridge treats the
    chaser's own reference orbit as chief so the virtual target's motion
    stays a pure analytic CW propagation independent of the chaser's actual
    (KSP-simulated, non-Keplerian-perturbed) trajectory.

    Parameters
    ----------
    chaser_state       : (14,) ndarray  Chaser's own [dq | dv] from KSP telemetry
                          (absolute pose in the vessel's orbital frame).
    target_state_lvlh   : (6,) ndarray  Virtual target's [x,y,z,vx,vy,vz]
                          relative to the chaser's reference orbit, LVLH.

    Returns
    -------
    rel_state : (14,) ndarray  Relative [dq | dv] of chaser w.r.t. target,
        in the same layout dynamics.py's SpacecraftModel expects.
    """
    chaser_pos, chaser_att = dq_to_pose(chaser_state[:8])
    chaser_omega = chaser_state[8:11]
    chaser_vel = chaser_state[11:14]

    target_pos = target_state_lvlh[0:3]
    target_vel = target_state_lvlh[3:6]

    rel_pos = chaser_pos - target_pos
    rel_vel = chaser_vel - target_vel
    # Target attitude/angular-velocity are not modeled by the CW point-mass
    # propagation (guidance.py's target is a translational-only reference);
    # relative attitude is therefore taken directly as the chaser's own
    # attitude/omega, consistent with treating the virtual target as
    # non-rotating in the LVLH frame -- an explicit simplification, to be
    # revisited once/if the target gains its own attitude model.
    rel_dq = dq_from_pose(rel_pos, chaser_att)
    rel_dv = np.concatenate([chaser_omega, rel_vel])
    return np.concatenate([rel_dq, rel_dv])


def synthesize_pose_measurement(rel_state: np.ndarray, config: PlantConfig,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Produce a noisy [r_vec | q_att] pose measurement of the true relative
    state, standing in for the LIDAR+optical perception stack's output
    until that module is implemented. Position noise is additive Gaussian;
    attitude noise is a small-angle random rotation composed onto the true
    attitude, which keeps the perturbed quaternion properly normalized
    (unlike additive quaternion noise, which does not).

    Parameters
    ----------
    rel_state : (14,) ndarray        True relative [dq | dv].
    config    : PlantConfig           Supplies the noise standard deviations.
    rng       : np.random.Generator   Seeded RNG for reproducibility.

    Returns
    -------
    z : (7,) ndarray  Noisy measurement [r_vec | q_att].
    """
    r_true, q_true = dq_to_pose(rel_state[:8])
    r_meas = r_true + rng.normal(0.0, config.pos_noise_std, size=3)

    theta = rng.normal(0.0, config.att_noise_std_rad, size=3)
    half = theta / 2.0
    dq_small = np.array([1.0, half[0], half[1], half[2]])
    dq_small = dq_small / np.linalg.norm(dq_small)
    q_meas = quat_mult(dq_small, q_true)
    q_meas = q_meas / np.linalg.norm(q_meas)

    return np.concatenate([r_meas, q_meas])


# ===========================================================================
# CATEGORY: Main control loop
# ===========================================================================

def run(config: PlantConfig,
        compute_control: Optional[Callable[[NavigationState, float], tuple]] = None,
        seed: int = 0,
        use_ariane5_chaser: bool = True):
    """
    Run the closed-loop KSP <-> GNC bridge.

    Parameters
    ----------
    config          : PlantConfig
        Bridge configuration (see PlantConfig docstring).
    compute_control : callable(nav_state, t) -> (u_force(3,), u_torque(3,)), optional
        Hook for the future control.py LTV-LQR controller. Receives the
        current IEKF NavigationState estimate and elapsed sim time, must
        return body-frame force [N] and torque [N*m] commands. Defaults to
        control.py's LTVLQRController tracking the origin.
    seed            : int
        RNG seed for the synthetic measurement noise.
    use_ariane5_chaser : bool
        If True (default), apply ariane5_chaser_params() for the duration
        of the run so the GNC stack's internal physics (MASS, INERTIA,
        THRUST_MAX, ...) matches an Ariane-5-derived chaser instead of
        dynamics.py's default 260 kg Starlink-class bus. This MUST match
        whatever vessel is actually flying in KSP -- the estimator and
        controller compute everything (drag, thrust saturation, Riccati
        gain, IEKF process-noise propagation) from these constants, not
        from anything KSP reports about the vessel's real mass/inertia.
        Set to False only if the KSP vessel genuinely matches the default
        260 kg / 0.05 N-thruster parameters in dynamics.py.

    Notes
    -----
    Loop timing budget: each cycle does one kRPC read, one IEKF predict+
    update (a handful of 14x14 linear-algebra ops plus finite-difference
    Jacobians), one virtual-target CW propagation, and one kRPC write. This
    directly measures the "tempo di esecuzione per ciclo di controllo"
    (control-cycle execution time) called for in objective #5; per-cycle
    wall time is printed so it can be logged/aggregated externally.
    """
    previous_params = apply_ariane5_chaser_params() if use_ariane5_chaser else None
    try:
        if compute_control is None:
            # Default: the LTV-LQR + keep-out barrier + attitude-PD
            # controller from control.py, tracking the origin (zero
            # relative state) as the nominal rendezvous target. Built
            # AFTER the Ariane-5 patch above so its internally-cached
            # gain (solve_lqr_gain_steady_state) and B(t)=1/mass matrix
            # reflect the correct chaser mass/inertia/thrust limit.
            _ctrl_holder = {"ctrl": LTVLQRController(ControllerConfig(), nominal_state0=np.zeros(6))}

            def compute_control(nav_state, t):
                return _ctrl_holder["ctrl"](nav_state.x, t)

        rng = np.random.default_rng(seed)
        plant = KSPPlant(config)

        t0 = plant.universal_time()
        target_state = config.target_offset0.copy()

        chaser_abs0 = plant.read_chaser_dq_state()
        rel0 = relative_state_from_absolute(chaser_abs0, target_state)
        nav = NavigationState(x=rel0, P=np.eye(14) * 1e-2)

        cycle = 0
        while config.max_cycles is None or cycle < config.max_cycles:
            cycle_start_wall = time.perf_counter()

            chaser_abs = plant.read_chaser_dq_state()
            target_state = cw_propagate_state(config.target_offset0,
                                              plant.universal_time() - t0)
            rel_true = relative_state_from_absolute(chaser_abs, target_state)

            z = synthesize_pose_measurement(rel_true, config, rng)

            u_force, u_torque = compute_control(nav, plant.universal_time() - t0)

            nav = iekf_step(nav, dt=config.dt, z=z,
                            u_force=u_force, u_torque=u_torque,
                            altitude_m=ALT_REF)

            plant.send_control(u_force, u_torque)

            cycle_wall_time = time.perf_counter() - cycle_start_wall
            r_est, _ = dq_to_pose(nav.x[:8])
            r_true, _ = dq_to_pose(rel_true[:8])
            print(f"[cycle {cycle:5d}] t={plant.universal_time() - t0:8.1f}s  "
                  f"|r_true|={np.linalg.norm(r_true):8.2f}m  "
                  f"|r_est|={np.linalg.norm(r_est):8.2f}m  "
                  f"cycle_time={cycle_wall_time * 1e3:6.2f}ms")

            cycle += 1
            sleep_remaining = config.dt - cycle_wall_time
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)
    finally:
        if previous_params is not None:
            restore_chaser_params(previous_params)


# ===========================================================================
# CATEGORY: Offline closed-loop simulation (no KSP / no kRPC required)
# ===========================================================================
#
# Everything above this point is the KSP hardware-in-the-loop bridge. The
# section below is a self-contained, pure-Python closed loop that exercises
# the exact same GNC stack (dynamics.propagate + navigation.iekf_step +
# control.LTVLQRController) without needing a running KSP instance, so the
# pipeline's convergence can be verified and plotted directly. It also
# supports one or more virtual "constellation" satellites (in addition to
# the collision-avoidance target) purely for trajectory visualization.
#
# Chaser vehicle model: by request, this section re-parameterizes the
# chaser as an Ariane-5-derived object rather than the 260 kg Starlink-class
# bus dynamics.py otherwise assumes. Since MASS / INERTIA / INERTIA_INV /
# AREA_REF / THRUST_MAX are consumed as hardcoded module-level globals
# inside dynamics.dynamics_rhs / dynamics.propagate / dynamics.accel_drag /
# dynamics.accel_srp (not passed as function arguments), the only way to
# swap the physical model without forking dynamics.py is to monkey-patch
# those globals -- in BOTH dynamics and control, since control.py imported
# MASS/INERTIA/INERTIA_INV/THRUST_MAX by value at its own import time and
# therefore holds its own separate references. ariane5_chaser_params()
# below documents the source numbers; apply_ariane5_chaser_params() performs
# the patch and returns the previous values so it can be undone.
# ===========================================================================

import dynamics as _dynamics_mod
import control as _control_mod
from dynamics import SpacecraftModel, SAFE_RADIUS


def ariane5_chaser_params() -> dict:
    """
    Physical parameters for an Ariane 5 ES upper-stage-class chaser
    (an EPS storable-propellant stage, representative of an ATV-derived or
    derelict-upper-stage rendezvous target/chaser), in place of dynamics.py's
    default 260 kg Starlink-class bus.

    Source figures: ESA's Ariane 5 EPS stage is ~3.35 m tall, ~3.94 m base
    diameter tapering to ~2.62 m, dry mass ~1,275 kg. Inertia is estimated
    from a solid-cylinder-shell approximation using that geometry, since no
    published inertia tensor exists for the EPS. Thruster: the Aestus main
    engine (27.5 kN) is far too powerful for fine proximity-ops control, so
    THRUST_MAX here instead models a small monopropellant RCS thruster
    cluster (representative of ATV's own 20x220 N + 8x490 N RCS thrusters,
    scaled down to a single-axis linearized command authority) rather than
    the main engine.

    Returns
    -------
    params : dict  Keys: mass, area_ref, i_xx, i_yy, i_zz, thrust_max, isp.
    """
    m = 1275.0            # [kg] dry mass (ESA EPS stage)
    radius = 0.5 * (3.94 + 2.62) / 2.0   # [m] mean radius of tapered body
    length = 3.35          # [m]
    i_axial = m * radius ** 2                                   # spin axis
    i_transverse = m * (3 * radius ** 2 + length ** 2) / 12.0    # thin shell approx
    return dict(
        mass=m,
        area_ref=np.pi * radius ** 2,       # [m^2] cross-section for drag
        i_xx=i_transverse,
        i_yy=i_transverse,
        i_zz=i_axial,
        thrust_max=2.0,      # [N] representative small monoprop RCS thruster
        isp=220.0,           # [s] typical hydrazine monoprop RCS Isp
    )


def apply_ariane5_chaser_params() -> dict:
    """
    Monkey-patch dynamics.py's (and control.py's already-imported copies of)
    MASS / INERTIA / INERTIA_INV / AREA_REF / THRUST_MAX / ISP to the
    Ariane-5-derived chaser parameters from ariane5_chaser_params().

    Returns
    -------
    previous : dict  The pre-patch values, so restore_chaser_params(previous)
        can undo this.
    """
    p = ariane5_chaser_params()
    inertia = np.diag([p["i_xx"], p["i_yy"], p["i_zz"]])
    inertia_inv = np.linalg.inv(inertia)

    previous = dict(
        mass=_dynamics_mod.MASS,
        area_ref=_dynamics_mod.AREA_REF,
        inertia=_dynamics_mod.INERTIA,
        inertia_inv=_dynamics_mod.INERTIA_INV,
        thrust_max=_dynamics_mod.THRUST_MAX,
        isp=_dynamics_mod.ISP,
    )

    for mod in (_dynamics_mod, _control_mod):
        mod.MASS = p["mass"]
        mod.THRUST_MAX = p["thrust_max"]
    _dynamics_mod.AREA_REF = p["area_ref"]
    _dynamics_mod.INERTIA = inertia
    _dynamics_mod.INERTIA_INV = inertia_inv
    _dynamics_mod.ISP = p["isp"]
    if hasattr(_control_mod, "INERTIA"):
        _control_mod.INERTIA = inertia
        _control_mod.INERTIA_INV = inertia_inv

    return previous


def restore_chaser_params(previous: dict):
    """Undo apply_ariane5_chaser_params() using its returned dict."""
    for mod in (_dynamics_mod, _control_mod):
        mod.MASS = previous["mass"]
        mod.THRUST_MAX = previous["thrust_max"]
    _dynamics_mod.AREA_REF = previous["area_ref"]
    _dynamics_mod.INERTIA = previous["inertia"]
    _dynamics_mod.INERTIA_INV = previous["inertia_inv"]
    _dynamics_mod.ISP = previous["isp"]
    if hasattr(_control_mod, "INERTIA"):
        _control_mod.INERTIA = previous["inertia"]
        _control_mod.INERTIA_INV = previous["inertia_inv"]


@dataclass
class OfflineSimConfig:
    """
    Configuration for the offline (no-KSP) closed-loop simulation.

    Attributes
    ----------
    dt              : float   Simulation / control step [s].
    n_steps         : int     Number of steps to propagate.
    r0, v0          : (3,)    Initial relative position [m] / velocity [m/s]
                               of the chaser w.r.t. the collision-avoidance
                               target, LVLH frame.
    q0              : (4,)    Initial relative attitude quaternion [w,x,y,z].
    omega0          : (3,)    Initial relative angular velocity [rad/s].
    pos_noise_std   : float   Synthetic position measurement noise std [m].
    att_noise_std   : float   Synthetic small-angle attitude noise std [rad].
    n_virtual_sats  : int     Number of extra virtual constellation satellites
                               to propagate (CW free-drift, for plotting only
                               -- they are not part of the estimation/control
                               loop, unlike the single collision-avoidance
                               target the IEKF/LQR track).
    seed            : int     RNG seed.
    """
    dt: float = 2.0
    n_steps: int = 5400          # 3 hours at dt=2s
    r0: np.ndarray = field(default_factory=lambda: np.array([100.0, -300.0, 20.0]))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q0: np.ndarray = field(default_factory=lambda: np.array([0.9998, 0.02, 0.0, 0.0]))
    omega0: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.001, 0.0005]))
    pos_noise_std: float = 0.5
    att_noise_std: float = 0.01
    n_virtual_sats: int = 3
    seed: int = 0


def simulate_closed_loop(cfg: OfflineSimConfig, use_ariane5_chaser: bool = True):
    """
    Run the full dynamics -> IEKF -> LTV-LQR closed loop offline (no KSP),
    logging everything needed for the convergence and trajectory plots.

    Parameters
    ----------
    cfg                 : OfflineSimConfig
    use_ariane5_chaser  : bool  If True, apply ariane5_chaser_params() for
                                 the duration of the run and restore the
                                 original (Starlink-class) parameters after.

    Returns
    -------
    log : dict  Time series and metadata (see keys inline below).
    """
    previous_params = apply_ariane5_chaser_params() if use_ariane5_chaser else None
    try:
        rng = np.random.default_rng(cfg.seed)

        q0 = cfg.q0 / np.linalg.norm(cfg.q0)
        dq0 = dq_from_pose(cfg.r0, q0)
        sc = SpacecraftModel(dq=dq0, dv=np.concatenate([cfg.omega0, cfg.v0]))

        x0 = np.concatenate([dq0, np.concatenate([cfg.omega0, cfg.v0])])
        nav = NavigationState(x=x0, P=np.eye(14) * 1e-2)

        # Track the origin (zero relative state) as the nominal rendezvous
        # target -- NOT the free CW drift of r0/v0. Using generate_nominal_
        # trajectory(state0=[r0,v0], ...) as the tracked reference (as a
        # naive first wiring of guidance.py + control.py would do) makes the
        # controller chase the *uncontrolled ballistic drift* of the initial
        # offset, which itself diverges under CW secular drift -- that is
        # not a rendezvous profile, it is the absence of one. Driving toward
        # the origin is what actually closes the range to the target.
        controller = LTVLQRController(ControllerConfig(), nominal_state0=np.zeros(6))

        # Virtual constellation satellites: independent CW free-drift states
        # (for visualization only), spread out from the chaser's reference
        # orbit at t=0.
        rng_virtual = np.random.default_rng(cfg.seed + 1)
        virtual_state0 = [
            np.concatenate([
                rng_virtual.uniform(-500, 500, size=3),
                rng_virtual.uniform(-0.05, 0.05, size=3),
            ])
            for _ in range(cfg.n_virtual_sats)
        ]

        t_hist = np.zeros(cfg.n_steps)
        pos_true_hist = np.zeros((cfg.n_steps, 3))
        pos_est_hist = np.zeros((cfg.n_steps, 3))
        att_err_deg_hist = np.zeros(cfg.n_steps)
        u_force_hist = np.zeros((cfg.n_steps, 3))
        u_torque_hist = np.zeros((cfg.n_steps, 3))
        virtual_pos_hist = np.zeros((cfg.n_steps, cfg.n_virtual_sats, 3))

        for k in range(cfg.n_steps):
            t = k * cfg.dt

            u_force, u_torque = controller(nav.x, t)

            sc = _dynamics_mod.propagate(
                sc, cfg.dt, u_force=u_force, u_torque=u_torque, altitude_m=ALT_REF)

            r_true, q_true = dq_to_pose(sc.dq)
            r_meas = r_true + rng.normal(0.0, cfg.pos_noise_std, size=3)
            theta = rng.normal(0.0, cfg.att_noise_std, size=3)
            half = theta / 2.0
            dq_small = np.array([1.0, half[0], half[1], half[2]])
            dq_small /= np.linalg.norm(dq_small)
            q_meas = quat_mult(dq_small, q_true)
            q_meas /= np.linalg.norm(q_meas)
            z = np.concatenate([r_meas, q_meas])

            nav = iekf_step(nav, dt=cfg.dt, z=z, u_force=u_force,
                            u_torque=u_torque, altitude_m=ALT_REF)

            r_est, _ = dq_to_pose(nav.x[:8])

            for j in range(cfg.n_virtual_sats):
                virtual_state0[j] = cw_propagate_state(virtual_state0[j], cfg.dt)
                virtual_pos_hist[k, j] = virtual_state0[j][:3]

            q_err = q_true.copy()
            if q_err[0] < 0.0:
                q_err = -q_err
            att_err_deg_hist[k] = np.degrees(2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0)))

            t_hist[k] = t
            pos_true_hist[k] = r_true
            pos_est_hist[k] = r_est
            u_force_hist[k] = u_force
            u_torque_hist[k] = u_torque

        return dict(
            t=t_hist, pos_true=pos_true_hist, pos_est=pos_est_hist,
            att_err_deg=att_err_deg_hist, u_force=u_force_hist,
            u_torque=u_torque_hist, virtual_pos=virtual_pos_hist,
            n_virtual_sats=cfg.n_virtual_sats,
        )
    finally:
        if previous_params is not None:
            restore_chaser_params(previous_params)


def plot_convergence(log: dict, save_path: str = None):
    """
    Plot three diagnostic figures from a simulate_closed_loop() log:
        1. Translational tracking error |r_true| (and estimator error
           |r_true - r_est|) vs time.
        2. Rotational (attitude) tracking error, in degrees, vs time.
        3. 3D trajectory of the chaser (true + estimated) relative to the
           collision-avoidance target at the origin, together with the
           tracked paths of the virtual constellation satellites.

    Parameters
    ----------
    log       : dict   Output of simulate_closed_loop().
    save_path : str, optional  If given, saves the figure to this path
                                (e.g. 'convergence.png') in addition to
                                returning the Figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D proj.)

    t_min = log["t"] / 60.0
    pos_err = np.linalg.norm(log["pos_true"], axis=1)
    est_err = np.linalg.norm(log["pos_true"] - log["pos_est"], axis=1)

    fig = plt.figure(figsize=(15, 10))

    # --- (1) Translational error ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_min, pos_err, label="|r_true| (range to target)", color="tab:blue")
    ax1.plot(t_min, est_err, label="|r_true - r_est| (nav. error)",
             color="tab:orange", linestyle="--")
    ax1.axhline(SAFE_RADIUS, color="tab:red", linestyle=":", label="keep-out radius")
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("Distance [m]")
    ax1.set_title("Translational tracking / estimation error")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- (2) Rotational (attitude) error ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_min, log["att_err_deg"], color="tab:green")
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("Attitude error [deg]")
    ax2.set_title("Rotational (attitude) tracking error")
    ax2.grid(True, alpha=0.3)

    # --- (3) 3D trajectory: chaser + virtual constellation satellites ---
    ax3 = fig.add_subplot(2, 2, (3, 4), projection="3d")
    ax3.plot(log["pos_true"][:, 0], log["pos_true"][:, 1], log["pos_true"][:, 2],
             color="tab:blue", label="Chaser (true)")
    ax3.plot(log["pos_est"][:, 0], log["pos_est"][:, 1], log["pos_est"][:, 2],
             color="tab:orange", linestyle="--", linewidth=0.8, label="Chaser (IEKF est.)")
    ax3.scatter(*log["pos_true"][0], color="tab:blue", marker="o", s=40, label="Chaser start")
    ax3.scatter(0, 0, 0, color="tab:red", marker="*", s=120, label="Target (keep-out center)")

    colors = plt.cm.viridis(np.linspace(0, 1, log["n_virtual_sats"]))
    for j in range(log["n_virtual_sats"]):
        traj = log["virtual_pos"][:, j, :]
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[j],
                 alpha=0.7, linewidth=1.0, label=f"Virtual sat {j+1}")
        ax3.scatter(*traj[-1], color=colors[j], marker="^", s=30)

    ax3.set_xlabel("Radial x [m]")
    ax3.set_ylabel("Along-track y [m]")
    ax3.set_zlabel("Cross-track z [m]")
    ax3.set_title("Relative trajectories (LVLH frame): chaser + virtual constellation")
    ax3.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    if KRPC_AVAILABLE:
        cfg = PlantConfig(
            vessel_name=None,  # use whichever vessel is active in KSP
            dt=1.0,
            target_offset0=np.array([100.0, -300.0, 20.0, 0.0, 0.0, 0.0]),
            pos_noise_std=0.5,
            att_noise_std_rad=0.01,
            max_cycles=None,
        )
        run(cfg)
    else:
        # No KSP/kRPC available in this environment: run the offline
        # closed-loop simulation (Ariane-5-class chaser) and produce the
        # convergence + trajectory plots instead.
        print("krpc not available -- running offline closed-loop simulation "
              "with an Ariane-5-derived chaser instead of the KSP bridge.")
        sim_cfg = OfflineSimConfig()
        sim_log = simulate_closed_loop(sim_cfg, use_ariane5_chaser=True)
        fig = plot_convergence(sim_log, save_path="gnc_convergence.png")
        print("Saved plot to gnc_convergence.png")