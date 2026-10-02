"""Diagnostic puzzle oracle ported from dyna-mpo/envs/ogbench_puzzle_oracle.py.

Uses privileged simulator state and OGBench's ButtonMarkovOracle, not a learned
policy. This module is deliberately separate from online training collection.
"""

from collections.abc import Mapping
from typing import Any, Protocol, cast

import gymnasium
import jax
import numpy as np
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle

from envs.ogbench_puzzle_mjx import OGBenchPuzzle3x3


class _GymPuzzleEnv(Protocol):
    _button_site_ids: list[int]
    _data: Any

    def compute_ob_info(self) -> Mapping[str, Any]: ...


class _OracleEnvAdapter:
    def __init__(self, arm_sampling_bounds: np.ndarray):
        self.unwrapped = self
        self._arm_sampling_bounds = np.asarray(arm_sampling_bounds, dtype=np.float64)


def _toggle_matrix_3x3() -> np.ndarray:
    matrix = np.zeros((9, 9), dtype=np.int32)
    for row in range(3):
        for col in range(3):
            pressed = row * 3 + col
            for drow, dcol in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor_row = row + drow
                neighbor_col = col + dcol
                if 0 <= neighbor_row < 3 and 0 <= neighbor_col < 3:
                    matrix[pressed, neighbor_row * 3 + neighbor_col] = 1
    return matrix


def solve_puzzle_3x3_presses(
    button_states: np.ndarray,
    target_button_states: np.ndarray,
) -> list[int]:
    """Return a deterministic minimal 3x3 Lights Out press sequence."""
    button_states = np.asarray(button_states, dtype=np.int32).reshape(9)
    target_button_states = np.asarray(target_button_states, dtype=np.int32).reshape(9)
    delta = (target_button_states - button_states) % 2
    toggle_matrix = _toggle_matrix_3x3()

    best_sequence: list[int] | None = None
    for mask in range(1 << 9):
        presses = np.array([(mask >> idx) & 1 for idx in range(9)], dtype=np.int32)
        if np.array_equal((presses @ toggle_matrix) % 2, delta):
            sequence = [idx for idx, should_press in enumerate(presses) if should_press]
            if best_sequence is None or len(sequence) < len(best_sequence):
                best_sequence = sequence

    if best_sequence is None:
        raise ValueError(
            "No 3x3 puzzle press solution for "
            f"button_states={button_states.tolist()} "
            f"target_button_states={target_button_states.tolist()}"
        )
    return best_sequence


def shape_oracle_action(
    action: np.ndarray,
    *,
    action_scale: float = 1.0,
    gripper_action: float | None = None,
) -> np.ndarray:
    shaped = np.asarray(action, dtype=np.float32).copy()
    shaped[:4] *= np.float32(action_scale)
    if gripper_action is not None:
        shaped[4] = np.float32(gripper_action)
    return np.clip(shaped, -1.0, 1.0).astype(np.float32)


class PuzzleButtonOracle:
    """Puzzle-level adapter that uses OGBench's single-button Markov oracle."""

    def __init__(
        self,
        *,
        arm_sampling_bounds: np.ndarray,
        seed: int = 0,
        max_step: int = 100,
        min_norm: float = 0.4,
    ):
        self._rng = np.random.default_rng(seed)
        self._adapter_env = _OracleEnvAdapter(arm_sampling_bounds)
        self._button_oracle = ButtonMarkovOracle(
            env=self._adapter_env,
            max_step=max_step,
            min_norm=min_norm,
        )
        self._target_button: int | None = None
        self._target_button_state: int | None = None
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def target_button(self) -> int | None:
        return self._target_button

    def reset(self) -> None:
        self._target_button = None
        self._target_button_state = None
        self._done = False

    def select_action(
        self,
        ob: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        button_states = np.asarray(info["button_states"], dtype=np.int32).reshape(9)
        target_button_states = np.asarray(info["target_button_states"], dtype=np.int32).reshape(9)

        if self._target_button is not None and self._target_button_state is not None:
            current_target_state = int(button_states[self._target_button])
            if current_target_state == self._target_button_state:
                self._target_button = None
                self._target_button_state = None

        if self._target_button is None:
            if np.array_equal(button_states, target_button_states):
                self._done = True
                return np.zeros(5, dtype=np.float32)

            press_sequence = solve_puzzle_3x3_presses(button_states, target_button_states)
            self._target_button = press_sequence[0]
            self._target_button_state = int(1 - button_states[self._target_button])
            oracle_info = self._button_oracle_info(info)
            self._button_oracle.reset(ob, oracle_info)
            self._button_oracle._final_pos = self._rng.uniform(  # noqa: SLF001
                *self._adapter_env._arm_sampling_bounds
            )
            self._button_oracle._final_yaw = self._rng.uniform(-np.pi, np.pi)  # noqa: SLF001
            self._done = False

        action = self._button_oracle.select_action(ob, self._button_oracle_info(info))
        return np.asarray(action, dtype=np.float32)

    def _button_oracle_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        if self._target_button is None or self._target_button_state is None:
            raise RuntimeError("Button oracle target requested before target selection.")
        oracle_info = dict(info)
        oracle_info["privileged/target_button"] = self._target_button
        oracle_info["privileged/target_button_state"] = self._target_button_state
        oracle_info["privileged/target_button_top_pos"] = np.asarray(
            oracle_info[f"privileged/button_{self._target_button}_top_pos"],
            dtype=np.float64,
        )
        return oracle_info


def _target_button_states_from_env(env: Any) -> np.ndarray:
    target = getattr(env, "_target_button_states", None)
    if target is not None:
        return np.asarray(target, dtype=np.int32).copy()
    cur_task_info = getattr(env, "cur_task_info", None)
    if isinstance(cur_task_info, Mapping) and "goal_button_states" in cur_task_info:
        return np.asarray(cur_task_info["goal_button_states"], dtype=np.int32).copy()
    raise AttributeError("Could not extract target button states from OGBench puzzle env.")


def gym_puzzle_oracle_info(env: gymnasium.Env) -> dict[str, Any]:
    unwrapped = cast(_GymPuzzleEnv, env.unwrapped)
    info = dict(unwrapped.compute_ob_info())
    info["target_button_states"] = _target_button_states_from_env(unwrapped)
    for button_idx, site_id in enumerate(unwrapped._button_site_ids):
        info[f"privileged/button_{button_idx}_top_pos"] = unwrapped._data.site_xpos[site_id].copy()
    return info


def _yaw_from_mat_np(mat: np.ndarray) -> float:
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def mjx_puzzle_oracle_info(
    env: OGBenchPuzzle3x3,
    state: Any,
) -> dict[str, Any]:
    # Avoid copying Warp's contact/constraint workspaces to the host each step.
    site_xpos, site_xmat, qpos = jax.device_get(
        (state.data.site_xpos, state.data.site_xmat, state.data.qpos)
    )
    button_states = np.asarray(jax.device_get(state.info["button_states"]), dtype=np.int32)
    target_button_states = np.asarray(
        jax.device_get(state.info["target_button_states"]), dtype=np.int32
    )

    info: dict[str, Any] = {
        "button_states": button_states,
        "target_button_states": target_button_states,
        "proprio/effector_pos": np.asarray(site_xpos[env._pinch_site_id]).copy(),
        "proprio/effector_yaw": np.array(
            [_yaw_from_mat_np(np.asarray(site_xmat[env._pinch_site_id]))]
        ),
        "proprio/gripper_opening": np.array(
            [np.clip(qpos[env._gripper_opening_qposadr] / 0.8, 0.0, 1.0)]
        ),
    }
    for button_idx, site_id in enumerate(env._button_site_ids):
        info[f"privileged/button_{button_idx}_state"] = int(button_states[button_idx])
        info[f"privileged/button_{button_idx}_top_pos"] = np.asarray(site_xpos[site_id]).copy()
    return info
