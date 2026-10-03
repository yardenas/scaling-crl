"""MJX puzzle simulator ported from dyna-mpo/envs/ogbench_puzzle_mjx.py.

The source port's simplified collision geometry, controller, observation and
reward semantics are retained. Training/autoreset live in puzzle.py.
"""

import re
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import ogbench  # noqa: F401
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import smooth
from mujoco_playground._src import mjx_env

from envs import ur5e_analytic_ik

_SUPPORTED_PUZZLE_3X3_RE = re.compile(r"^puzzle-3x3-(?:play-)?singletask-task([1-5])-v0$")
_DEFAULT_ENV_NAME = "puzzle-3x3-singletask-task4-v0"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        env_name=_DEFAULT_ENV_NAME,
        target_button_states=None,
        button_start_probability=0.0,
        button_start_height=0.04,
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=500,
        impl="jax",
        graph_mode=None,
        solver=None,
        solver_iterations=5,
        solver_ls_iterations=1,
        integrator=None,
        cone=None,
        normal_only_contacts=False,
        naconmax=2048,
        njmax=1024,
        sparse=False,
        terminate_at_goal=True,
        nan_termination_penalty=-10.0,
        ik_solver="analytic",
        diff_ik_iters=20,
        diff_ik_damping=1e-12,
        diff_ik_max_angle_change=np.deg2rad(45.0),
        gripper_force_limit=12.75,
        gripper_limit_contact_threshold=0.7718,
        gripper_limit_contact_gain=32.5,
        gripper_limit_contact_max=0.38,
    )


def _resolve_warp_graph_mode(graph_mode: Any) -> Any:
    if not isinstance(graph_mode, str):
        return graph_mode

    graph_mode_enums = []
    import mujoco.mjx.warp as mjxw

    for candidate in (
        getattr(mjxw, "GraphMode", None),
        getattr(getattr(mjxw, "types", None), "GraphMode", None),
    ):
        if candidate is not None:
            graph_mode_enums.append(candidate)

    try:
        from warp._src.jax_experimental import ffi as warp_ffi

        graph_mode_enums.append(warp_ffi.GraphMode)
    except ImportError:
        pass

    for graph_mode_enum in graph_mode_enums:
        if hasattr(graph_mode_enum, graph_mode):
            return getattr(graph_mode_enum, graph_mode)

    available_modes = sorted(
        {
            name
            for graph_mode_enum in graph_mode_enums
            for name in dir(graph_mode_enum)
            if name.isupper()
        }
    )
    available_text = ", ".join(available_modes) if available_modes else "none found"
    raise ValueError(
        f"Unsupported graph_mode={graph_mode!r} for this MuJoCo/MJX/Warp install. "
        f"Available modes: {available_text}."
    )


def _resolve_mujoco_enum(value: Any, enum_cls: Any, prefix: str, config_name: str) -> Any:
    if not isinstance(value, str):
        return value

    members = {
        name.upper(): getattr(enum_cls, name)
        for name in dir(enum_cls)
        if name.upper().startswith(f"{prefix}_")
    }
    normalized = value.upper()
    candidates = [normalized]
    if not normalized.startswith(f"{prefix}_"):
        candidates.append(f"{prefix}_{normalized}")

    for candidate in candidates:
        if candidate in members:
            return members[candidate]

    available = ", ".join(sorted(members))
    raise ValueError(f"Unsupported {config_name}={value!r}. Expected one of: {available}.")


def _apply_model_option_overrides(model: mujoco.MjModel, config: Any) -> None:
    if config.solver is not None:
        model.opt.solver = _resolve_mujoco_enum(
            config.solver,
            mujoco.mjtSolver,
            "MJSOL",
            "solver",
        )
    if config.integrator is not None:
        model.opt.integrator = _resolve_mujoco_enum(
            config.integrator,
            mujoco.mjtIntegrator,
            "MJINT",
            "integrator",
        )
    if config.cone is not None:
        model.opt.cone = _resolve_mujoco_enum(
            config.cone,
            mujoco.mjtCone,
            "MJCONE",
            "cone",
        )
    if config.solver_iterations is not None:
        model.opt.iterations = int(config.solver_iterations)
    if config.solver_ls_iterations is not None:
        model.opt.ls_iterations = int(config.solver_ls_iterations)


def _apply_contact_simplification(model: mujoco.MjModel, config: Any) -> None:
    if not config.normal_only_contacts:
        return
    if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError(
            "normal_only_contacts=True sets active contact geoms to condim=1, "
            "but this MJX JAX build does not support condim=1 with cone=ELLIPTIC."
        )
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        model.geom_condim[geom_id] = 1
        model.geom_friction[geom_id] = 0.0


_ACTION_LOW = np.array([-0.05, -0.05, -0.05, -0.3, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float32)
_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)


def is_puzzle_3x3_env_name(env_name: str) -> bool:
    return _SUPPORTED_PUZZLE_3X3_RE.match(env_name.replace("_", "-")) is not None


def canonical_puzzle_3x3_env_name(env_name: str) -> str:
    normalized = env_name.replace("_", "-")
    match = _SUPPORTED_PUZZLE_3X3_RE.match(normalized)
    if match is None:
        raise ValueError(
            "The local puzzle MJX backend supports puzzle-3x3 singletask task1-task5, "
            f"got {env_name!r}."
        )
    return normalized.replace("-play-singletask-", "-singletask-")


def puzzle_3x3_task_id(env_name: str) -> int:
    match = _SUPPORTED_PUZZLE_3X3_RE.match(env_name.replace("_", "-"))
    if match is None:
        raise ValueError(f"Unsupported puzzle-3x3 env_name={env_name!r}.")
    return int(match.group(1))


def _build_reference_env(env_name: str):
    env = gymnasium.make(canonical_puzzle_3x3_env_name(env_name))
    env.reset(seed=0)
    return env.unwrapped


def _sanitize_model_for_mjx(model: mujoco.MjModel) -> None:
    pad_collision_geoms = {
        "ur5e/robotiq/right_pad1",
        "ur5e/robotiq/right_pad2",
        "ur5e/robotiq/left_pad1",
        "ur5e/robotiq/left_pad2",
    }
    pad_contype, pad_conaffinity = 1, 2
    button_contype, button_conaffinity = 2, 1
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name
        body_name = model.body(model.geom_bodyid[geom_id]).name
        is_pad_collision = geom_name in pad_collision_geoms
        is_button_collision = body_name.startswith("button_") and model.geom_contype[geom_id] != 0
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        elif is_pad_collision:
            model.geom_contype[geom_id] = pad_contype
            model.geom_conaffinity[geom_id] = pad_conaffinity
        elif is_button_collision:
            model.geom_contype[geom_id] = button_contype
            model.geom_conaffinity[geom_id] = button_conaffinity
        else:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_CAPSULE


def _mat_to_quat(mat: jax.Array) -> jax.Array:
    qw = 0.5 * jnp.sqrt(jnp.maximum(0.0, 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]))
    qx = (
        0.5
        * jnp.sign(mat[2, 1] - mat[1, 2])
        * jnp.sqrt(jnp.maximum(0.0, 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]))
    )
    qy = (
        0.5
        * jnp.sign(mat[0, 2] - mat[2, 0])
        * jnp.sqrt(jnp.maximum(0.0, 1.0 - mat[0, 0] + mat[1, 1] - mat[2, 2]))
    )
    qz = (
        0.5
        * jnp.sign(mat[1, 0] - mat[0, 1])
        * jnp.sqrt(jnp.maximum(0.0, 1.0 - mat[0, 0] - mat[1, 1] + mat[2, 2]))
    )
    return mjx_math.normalize(jnp.array([qw, qx, qy, qz], dtype=jnp.float32))


def _yaw_from_mat(mat: jax.Array) -> jax.Array:
    return jnp.arctan2(mat[1, 0], mat[0, 0])


def _quat_from_z_radians(theta: jax.Array) -> jax.Array:
    half_theta = 0.5 * theta
    return jnp.array(
        [jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)],
        dtype=jnp.float32,
    )


class OGBenchPuzzle3x3(mjx_env.MjxEnv):
    """Planner-focused MJX port of OGBench puzzle-3x3 singletask envs."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(
            default_config() if config is None else config,
            config_overrides=config_overrides,
        )
        if self._config.ik_solver not in ("analytic", "diff"):
            raise ValueError(
                f"Unsupported ik_solver={self._config.ik_solver!r}; expected 'analytic' or 'diff'."
            )
        if self._config.impl == "warp" and self._config.ik_solver != "analytic":
            raise ValueError("Warp requires ik_solver='analytic'; differential IK uses JAX internals.")

        custom_target = self._config.target_button_states
        if custom_target is not None:
            custom_target = np.asarray(custom_target)
            if custom_target.shape != (9,) or not np.all((custom_target == 0) | (custom_target == 1)):
                raise ValueError("Puzzle target must contain exactly nine binary (0 or 1) button states in row order.")
            custom_target = custom_target.astype(np.int32)

        self._env_name = canonical_puzzle_3x3_env_name(str(self._config.env_name))
        ref_env = _build_reference_env(self._env_name)
        self._xml_path = f"ogbench:{self._env_name}"
        self._mj_model = ref_env._model
        _sanitize_model_for_mjx(self._mj_model)
        self._mj_model.opt.timestep = self.sim_dt
        _apply_model_option_overrides(self._mj_model, self._config)
        _apply_contact_simplification(self._mj_model, self._config)

        self._arm_joint_ids = np.asarray(ref_env._arm_joint_ids, dtype=np.int32)
        self._arm_qposadr = np.asarray(
            [self._mj_model.jnt_qposadr[joint_id] for joint_id in self._arm_joint_ids],
            dtype=np.int32,
        )
        self._arm_dofadr = np.asarray(
            [self._mj_model.jnt_dofadr[joint_id] for joint_id in self._arm_joint_ids],
            dtype=np.int32,
        )
        self._arm_actuator_ids = np.asarray(ref_env._arm_actuator_ids, dtype=np.int32)
        self._gripper_actuator_ids = np.asarray(ref_env._gripper_actuator_ids, dtype=np.int32)
        if self._config.gripper_force_limit is not None:
            gripper_force_limit = float(self._config.gripper_force_limit)
            self._mj_model.actuator_forcerange[self._gripper_actuator_ids, 0] = -gripper_force_limit
            self._mj_model.actuator_forcerange[self._gripper_actuator_ids, 1] = gripper_force_limit
        put_model_kwargs: dict[str, Any] = {"impl": self._config.impl}
        if self._config.graph_mode is not None:
            if self._config.impl != "warp":
                raise ValueError("graph_mode is only supported with impl='warp'.")
            put_model_kwargs["graph_mode"] = _resolve_warp_graph_mode(self._config.graph_mode)
        self._mjx_model = mjx.put_model(self._mj_model, **put_model_kwargs)

        self._gripper_opening_qposadr = int(
            self._mj_model.jnt_qposadr[ref_env._gripper_opening_joint_id]
        )
        self._pinch_site_id = int(ref_env._pinch_site_id)
        self._attach_site_id = int(ref_env._attach_site_id)
        self._attach_body_id = int(self._mj_model.site_bodyid[self._attach_site_id])
        self._right_pad_body_id = int(self._mj_model.body("ur5e/robotiq/right_pad").id)
        self._button_qposadr = np.asarray(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(f"buttonbox_joint_{i}").id]
                for i in range(9)
            ],
            dtype=np.int32,
        )
        self._button_dofadr = np.asarray(
            [
                self._mj_model.jnt_dofadr[self._mj_model.joint(f"buttonbox_joint_{i}").id]
                for i in range(9)
            ],
            dtype=np.int32,
        )
        self._button_site_ids = np.asarray(
            [self._mj_model.site(f"btntop_{i}").id for i in range(9)],
            dtype=np.int32,
        )
        self._toggle_matrix = self._make_toggle_matrix()

        self._init_qpos = np.asarray(ref_env._data.qpos, dtype=np.float32)
        self._init_qvel = np.asarray(ref_env._data.qvel, dtype=np.float32)
        self._init_ctrl = np.asarray(ref_env._data.ctrl, dtype=np.float32)
        self._init_button_states = np.asarray(ref_env._cur_button_states, dtype=np.int32)
        self._target_button_states = (
            np.asarray(ref_env._target_button_states, dtype=np.int32)
            if custom_target is None else custom_target
        )
        self._arm_sampling_bounds = np.asarray(ref_env._arm_sampling_bounds, dtype=np.float32)
        self._t_pa_quat = np.asarray(ref_env._T_pa.rotation().wxyz, dtype=np.float32)
        self._t_pa_pos = np.asarray(ref_env._T_pa.translation(), dtype=np.float32)
        self._ctrl_low = np.asarray(self._mj_model.actuator_ctrlrange[:, 0], dtype=np.float32)
        self._ctrl_high = np.asarray(self._mj_model.actuator_ctrlrange[:, 1], dtype=np.float32)
        if self._config.button_start_probability > 0:
            self._prepare_button_starts(ref_env)
        ref_env.close()

    def _prepare_button_starts(self, ref_env):
        """Cache nine physical reset poses above buttons, without pressing them."""
        from ogbench.manipspace import lie

        qpos, ctrl = [], []
        for site_id in self._button_site_ids:
            position = ref_env._data.site_xpos[site_id].copy()
            position[2] += self._config.button_start_height
            pose = lie.SE3.from_rotation_and_translation(ref_env._effector_down_rotation, position)
            attach = pose @ ref_env._T_pa
            joints = ref_env._ik.solve(pos=attach.translation(), quat=attach.rotation().wxyz,
                                      curr_qpos=ref_env._home_qpos)
            data = mujoco.MjData(self._mj_model)
            data.qpos[:] = self._init_qpos
            data.qpos[self._arm_qposadr] = joints
            data.ctrl[:] = self._init_ctrl
            data.ctrl[self._arm_actuator_ids] = joints
            mujoco.mj_forward(self._mj_model, data)
            if (np.linalg.norm(data.site_xpos[self._pinch_site_id] - position) > 0.015
                    or np.any(data.qpos[self._button_qposadr] <= -0.02)):
                raise ValueError("Button-start pose is not an unpressed hover; increase button_start_height.")
            qpos.append(data.qpos.copy())
            ctrl.append(data.ctrl.copy())
        self._button_start_qpos = jnp.asarray(np.asarray(qpos), dtype=jnp.float32)
        self._button_start_ctrl = jnp.asarray(np.asarray(ctrl), dtype=jnp.float32)

    def _make_toggle_matrix(self) -> jax.Array:
        matrix = np.zeros((9, 9), dtype=np.int32)
        for row in range(3):
            for col in range(3):
                pressed = row * 3 + col
                for drow, dcol in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor_row = row + drow
                    neighbor_col = col + dcol
                    if 0 <= neighbor_row < 3 and 0 <= neighbor_col < 3:
                        matrix[pressed, neighbor_row * 3 + neighbor_col] = 1
        return jnp.asarray(matrix)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 5

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def model_assets(self) -> dict[str, Any]:
        return {}

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos, qvel, ctrl = (jnp.asarray(self._init_qpos), jnp.asarray(self._init_qvel),
                           jnp.asarray(self._init_ctrl))
        button_start, start_button = jnp.asarray(False), jnp.asarray(-1, jnp.int32)
        if self._config.button_start_probability > 0:
            near_key, button_key = jax.random.split(rng)
            button_start = jax.random.bernoulli(near_key, self._config.button_start_probability)
            button = jax.random.randint(button_key, (), 0, 9)
            qpos = jnp.where(button_start, self._button_start_qpos[button], qpos)
            qvel = jnp.where(button_start, jnp.zeros_like(qvel), qvel)
            ctrl = jnp.where(button_start, self._button_start_ctrl[button], ctrl)
            start_button = jnp.where(button_start, button, -1)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos, qvel=qvel, ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        button_states = jnp.asarray(self._init_button_states, dtype=jnp.int32)
        info = {
            "rng": rng,
            "button_start": button_start,
            "start_button": start_button,
            "button_states": button_states,
            "prev_button_states": button_states,
            "prev_button_qpos": data.qpos[self._button_qposadr],
            "target_button_states": jnp.asarray(self._target_button_states, dtype=jnp.int32),
            "success": jnp.asarray(False),
            "ik_no_solution": jnp.asarray(False),
            "time_out": jnp.asarray(0.0, dtype=jnp.float32),
        }
        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2, dtype=jnp.float32)
        metrics = {
            "success": jnp.asarray(0.0, dtype=jnp.float32),
            "dense_reward": jnp.asarray(0.0, dtype=jnp.float32),
            "valid": jnp.asarray(1.0, dtype=jnp.float32),
            "nan_termination": jnp.asarray(0.0, dtype=jnp.float32),
            "ik_no_solution": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def complete_restored_info(
        self,
        planner_state: dict[str, Any],
        data: mjx.Data,
        info: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        saved_info = planner_state.get("info")
        if not isinstance(saved_info, dict):
            saved_info = {}

        def _state_leaf(name: str, default: jax.Array) -> jax.Array:
            value = saved_info.get(name, planner_state.get(name, default))
            return jnp.asarray(value, dtype=default.dtype)

        button_states = _state_leaf(
            "button_states",
            jnp.asarray(self._init_button_states, dtype=jnp.int32),
        )
        prev_button_states = _state_leaf("prev_button_states", button_states)
        target_button_states = _state_leaf(
            "target_button_states",
            jnp.asarray(self._target_button_states, dtype=jnp.int32),
        )
        completed = dict(info)
        completed["button_states"] = button_states
        completed["prev_button_states"] = prev_button_states
        completed["target_button_states"] = target_button_states
        completed["success"] = _state_leaf(
            "success", jnp.all(button_states == target_button_states)
        )
        completed["ik_no_solution"] = _state_leaf("ik_no_solution", jnp.asarray(False))
        completed["prev_button_qpos"] = _state_leaf(
            "prev_button_qpos",
            data.qpos[self._button_qposadr],
        )
        return completed

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self._step(state, action, update_obs=True)

    def planner_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self._step(state, action, update_obs=False)

    def _step(
        self,
        state: mjx_env.State,
        action: jax.Array,
        *,
        update_obs: bool,
    ) -> mjx_env.State:
        deltas = self._unnormalize_action(action)
        ctrl, ik_no_solution = self._control_from_action(state.data, deltas)
        prev_button_qpos = state.data.qpos[self._button_qposadr]
        prev_button_states = state.info["button_states"]
        target_button_states = state.info["target_button_states"]

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        button_states = self.update_button_states(
            prev_button_states,
            prev_button_qpos,
            data.qpos[self._button_qposadr],
        )
        info = dict(state.info)
        info["prev_button_qpos"] = prev_button_qpos
        info["prev_button_states"] = prev_button_states
        info["button_states"] = button_states
        info["ik_no_solution"] = ik_no_solution

        success = jnp.all(button_states == target_button_states)
        info["success"] = success
        dense_reward = self._dense_reward(button_states, target_button_states)
        reward = self._reward(button_states, target_button_states, success)
        nan_termination = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        nan_termination = nan_termination | ~jnp.isfinite(ctrl).all()
        dense_reward = jnp.where(
            nan_termination,
            jnp.asarray(self._config.nan_termination_penalty, dtype=jnp.float32),
            dense_reward,
        )
        reward = jnp.where(
            nan_termination,
            jnp.asarray(self._config.nan_termination_penalty, dtype=jnp.float32),
            reward,
        )
        done = jnp.logical_or(
            nan_termination,
            jnp.logical_and(success, bool(self._config.terminate_at_goal)),
        ).astype(jnp.float32)
        obs = self._get_obs(data, info) if update_obs else state.obs
        metrics = {
            "success": success.astype(jnp.float32),
            "dense_reward": dense_reward,
            "valid": jnp.logical_not(nan_termination).astype(jnp.float32),
            "nan_termination": nan_termination.astype(jnp.float32),
            "ik_no_solution": ik_no_solution.astype(jnp.float32),
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def update_button_states(
        self,
        prev_button_states: jax.Array,
        prev_button_qpos: jax.Array,
        current_button_qpos: jax.Array,
    ) -> jax.Array:
        crossings = jnp.logical_and(prev_button_qpos > -0.02, current_button_qpos <= -0.02)
        toggles = crossings.astype(jnp.int32) @ self._toggle_matrix
        return (prev_button_states.astype(jnp.int32) + toggles) % 2

    def _unnormalize_action(self, action: jax.Array) -> jax.Array:
        low = jnp.asarray(_ACTION_LOW)
        high = jnp.asarray(_ACTION_HIGH)
        return 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (high - low) + low

    def _control_from_action(
        self, data: mjx.Data, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        target_pos, target_quat = self._target_attach_pose(data, action)
        if self._config.ik_solver == "diff":
            qpos_target, ik_no_solution = self._solve_diff_ik_with_status(
                data, target_pos, target_quat
            )
        else:
            qpos_target, ik_no_solution = self._solve_analytic_ik_with_status(
                data, target_pos, target_quat
            )

        gripper_opening = jnp.clip(data.qpos[self._gripper_opening_qposadr] / 0.8, 0.0, 1.0)
        gripper_target = jnp.clip(gripper_opening + action[4], 0.0, 1.0)

        ctrl = data.ctrl
        ctrl = ctrl.at[self._arm_actuator_ids].set(qpos_target)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * gripper_target)
        ctrl = jnp.clip(ctrl, self._ctrl_low, self._ctrl_high)
        ctrl_nonfinite = jnp.logical_not(jnp.isfinite(ctrl).all())
        ctrl = jnp.where(ctrl_nonfinite, data.ctrl, ctrl)
        ik_no_solution = jnp.logical_or(ik_no_solution, ctrl_nonfinite)
        ik_no_solution = jnp.logical_or(ik_no_solution, jnp.logical_not(jnp.isfinite(ctrl).all()))
        return ctrl, ik_no_solution

    def _target_attach_pose(
        self,
        data: mjx.Data,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        effector_pos = data.site_xpos[self._pinch_site_id]
        effector_yaw = _yaw_from_mat(data.site_xmat[self._pinch_site_id])
        target_effector_pos = jnp.clip(
            effector_pos + action[:3],
            jnp.asarray([0.25, -0.35, 0.02], dtype=jnp.float32),
            jnp.asarray([0.6, 0.35, 0.35], dtype=jnp.float32),
        )
        target_yaw = jnp.clip(effector_yaw + action[3], -jnp.pi, jnp.pi)
        target_effector_quat = mjx_math.quat_mul(
            _quat_from_z_radians(target_yaw),
            jnp.asarray(_DOWN_QUAT),
        )
        target_attach_pos = target_effector_pos + mjx_math.rotate(
            jnp.asarray(self._t_pa_pos),
            target_effector_quat,
        )
        target_attach_quat = mjx_math.quat_mul(
            target_effector_quat,
            jnp.asarray(self._t_pa_quat),
        )
        return target_attach_pos, target_attach_quat

    def _solve_analytic_ik(
        self,
        data: mjx.Data,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> jax.Array:
        qpos_target, _ = self._solve_analytic_ik_with_status(data, target_pos, target_quat)
        return qpos_target

    def _solve_analytic_ik_with_status(
        self,
        data: mjx.Data,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        qpos0 = data.qpos[self._arm_qposadr]
        return ur5e_analytic_ik.solve_ik_with_status(qpos0, target_pos, target_quat)

    def _solve_diff_ik_with_status(
        self,
        data: mjx.Data,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        qpos0 = data.qpos[self._arm_qposadr]
        qpos_target = self._solve_diff_ik(data, target_pos, target_quat)
        no_soln = jnp.logical_not(jnp.isfinite(target_pos).all())
        no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(target_quat).all()))
        no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(qpos_target).all()))
        qpos_target = jnp.where(no_soln, qpos0, qpos_target)
        no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(qpos_target).all()))
        return qpos_target, no_soln

    def _solve_diff_ik(
        self,
        data: mjx.Data,
        target_pos: jax.Array,
        target_quat: jax.Array,
    ) -> jax.Array:
        qpos0 = data.qpos[self._arm_qposadr]

        def _scan_fn(qpos: jax.Array, _: Any):
            ik_data = data.replace(qpos=data.qpos.at[self._arm_qposadr].set(qpos))
            ik_data = smooth.kinematics(self._mjx_model, ik_data)

            current_pos = ik_data.site_xpos[self._attach_site_id]
            current_quat = _mat_to_quat(ik_data.site_xmat[self._attach_site_id])
            pos_error = target_pos - current_pos
            quat_error = mjx_math.quat_mul(target_quat, mjx_math.quat_inv(current_quat))
            axis, angle = mjx_math.quat_to_axis_angle(mjx_math.normalize(quat_error))
            rot_error = axis * angle

            joint_axes = ik_data.xaxis[self._arm_joint_ids]
            joint_anchors = ik_data.xanchor[self._arm_joint_ids]
            jacp = jnp.cross(joint_axes, current_pos - joint_anchors)
            jacr = joint_axes
            jac = jnp.concatenate(
                [jacp.T, jacr.T],
                axis=0,
            )
            error = jnp.concatenate([pos_error, rot_error], axis=0)
            damping = self._config.diff_ik_damping * jnp.eye(6, dtype=jnp.float32)
            hessian = jac @ jac.T + damping
            update = jac.T @ jnp.linalg.solve(hessian, error)
            max_update = jnp.max(jnp.abs(update))
            update = jnp.where(
                max_update > self._config.diff_ik_max_angle_change,
                update * self._config.diff_ik_max_angle_change / max_update,
                update,
            )
            return qpos + update, None

        qpos_target, _ = jax.lax.scan(
            _scan_fn,
            qpos0,
            (),
            length=int(self._config.diff_ik_iters),
        )
        return qpos_target

    def _reward(
        self,
        button_states: jax.Array,
        target_button_states: jax.Array,
        success: jax.Array,
    ) -> jax.Array:
        if bool(self._config.sparse):
            return jnp.where(success, 0.0, -1.0).astype(jnp.float32)
        return self._dense_reward(button_states, target_button_states)

    def _dense_reward(
        self,
        button_states: jax.Array,
        target_button_states: jax.Array,
    ) -> jax.Array:
        matches = jnp.sum(button_states == target_button_states)
        return (matches - button_states.shape[0]).astype(jnp.float32)

    def _get_obs(self, data: mjx.Data, info: dict[str, jax.Array]) -> jax.Array:
        joint_pos = data.qpos[self._arm_qposadr]
        joint_vel = data.qvel[self._arm_dofadr]
        effector_pos = (data.site_xpos[self._pinch_site_id] - jnp.asarray(_XYZ_CENTER)) * 10.0
        effector_yaw = _yaw_from_mat(data.site_xmat[self._pinch_site_id])
        gripper_opening = jnp.array(
            [jnp.clip(data.qpos[self._gripper_opening_qposadr] / 0.8, 0.0, 1.0) * 3.0],
            dtype=jnp.float32,
        )
        cfrc_contact = jnp.clip(
            jnp.linalg.norm(data._impl.cfrc_ext[self._right_pad_body_id]) / 50.0,
            0.0,
            1.0,
        )
        limit_contact = jnp.clip(
            (
                data.qpos[self._gripper_opening_qposadr]
                - float(self._config.gripper_limit_contact_threshold)
            )
            * float(self._config.gripper_limit_contact_gain),
            0.0,
            float(self._config.gripper_limit_contact_max),
        )
        gripper_contact = jnp.array([jnp.maximum(cfrc_contact, limit_contact)], dtype=jnp.float32)
        button_states = info["button_states"].astype(jnp.int32)
        button_one_hot = jnp.eye(2, dtype=jnp.float32)[button_states]
        button_pos = data.qpos[self._button_qposadr, None] * 120.0
        button_vel = data.qvel[self._button_dofadr, None]
        button_obs = jnp.concatenate([button_one_hot, button_pos, button_vel], axis=1).ravel()
        return jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                effector_pos,
                jnp.cos(effector_yaw).reshape(1),
                jnp.sin(effector_yaw).reshape(1),
                gripper_opening,
                gripper_contact,
                button_obs,
            ],
            axis=0,
        ).astype(jnp.float32)
