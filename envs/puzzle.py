"""Goal observations and batched episodic online rollouts for the MJX puzzle.

Layout: [achieved buttons (9), simulator observation (55), target buttons (9)].
The first nine state coordinates are replay's achieved-goal representation.
"""

from importlib.metadata import PackageNotFoundError, version

import jax
import jax.numpy as jnp


def check_runtime(impl):
    if impl not in ("jax", "warp"):
        raise ValueError("Puzzle backend must be 'jax' (debugging) or 'warp' (NVIDIA GPU).")
    try:
        compatible = all(tuple(map(int, version(p).split(".")[:2])) >= minimum
                         for p, minimum in (("mujoco", (3, 8)), ("mujoco-mjx", (3, 8)),
                                            ("jax", (0, 6))))
        for package in ("ogbench", "playground", "warp-lang"):
            version(package)
    except PackageNotFoundError:
        compatible = False
    if not compatible:
        raise RuntimeError("Puzzle dependencies are missing or outdated. Run 'uv sync --locked', "
                           "then 'uv run python main.py environment=puzzle_3x3'.")
    if impl == "warp" and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("MJX-Warp requires an NVIDIA CUDA GPU. Use backend=jax for CPU debugging.")


class PuzzleEnv:
    """Unbatched simulator with a goal-conditioned observation."""

    goal_size = 9
    state_size = 64
    observation_size = 73
    action_size = 5

    def __init__(self, env_name, impl="warp", sparse=True, config_overrides=None, target=None):
        check_runtime(impl)
        from envs.ogbench_puzzle_mjx import OGBenchPuzzle3x3

        overrides = dict(config_overrides or {})
        overrides.update(env_name=env_name, impl=impl, sparse=sparse,
                         target_button_states=target)
        self.simulator = OGBenchPuzzle3x3(config_overrides=overrides)

    @property
    def unwrapped(self):
        return self

    def _adapt(self, state):
        achieved = state.info["button_states"].astype(jnp.float32)
        target = state.info["target_button_states"].astype(jnp.float32)
        # A failed simulation terminates immediately. Keep its replay endpoint
        # finite so even a masked TD target cannot contaminate network gradients.
        obs = jnp.nan_to_num(jnp.concatenate((achieved, state.obs, target)),
                             nan=0.0, posinf=0.0, neginf=0.0)
        return state.replace(obs=obs, metrics={**state.metrics, "dist": jnp.sum(achieved != target).astype(jnp.float32)})

    def reset(self, key):
        return self._adapt(self.simulator.reset(key))

    def step(self, state, action):
        return self._adapt(self.simulator.step(state, action))


class PuzzleTrainingEnv:
    """Vmap + time limits + complete cached autoreset, compatible with scan.

    Like the source environment, resets return the fixed seed-0 task start.
    Both physical data and task info must reset: button bits aren't in mjx.Data.
    Preserve terminal reward/metrics and the pre-reset endpoint for SAC.
    """

    def __init__(self, env, episode_length):
        self.env = env
        self.episode_length = episode_length
        self.action_size = env.action_size
        self.goal_size = env.goal_size

    @property
    def unwrapped(self):
        return self.env

    def reset(self, keys):
        state = jax.vmap(self.env.reset)(keys)
        info = dict(state.info)
        info.update(first_data=state.data, first_obs=state.obs, first_info=state.info,
                    steps=jnp.zeros_like(state.done, dtype=jnp.int32),
                    truncation=jnp.zeros_like(state.done),
                    final_observation=jnp.zeros_like(state.obs),
                    final_observation_valid=jnp.zeros_like(state.done, dtype=bool))
        return state.replace(info=info)

    def step(self, state, actions):
        cached = state.info
        # Pass only the simulator's info through vmap; don't duplicate the cached
        # physics tree through every simulator substep.
        task_info = {name: cached[name] for name in cached["first_info"]}
        stepped = jax.vmap(self.env.step)(state.replace(info=task_info), actions)
        steps = jnp.where(state.done, 0, cached["steps"]) + 1
        timeout = steps >= self.episode_length
        terminal = stepped.done.astype(bool)
        done = timeout | terminal

        def select_reset(first, current):
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, first, current)

        info = jax.tree_util.tree_map(select_reset, cached["first_info"], stepped.info)
        info.update(first_data=cached["first_data"], first_obs=cached["first_obs"],
                    first_info=cached["first_info"], steps=steps,
                    truncation=(timeout & ~terminal).astype(jnp.float32),
                    final_observation=stepped.obs, final_observation_valid=done)
        if self.env.simulator.mjx_model.impl.value == "warp":
            from mujoco.mjx.warp.types import DATA_NON_VMAP, DataWarp

            def reset_physics(first, current):
                if isinstance(current, DataWarp):
                    # Warp's contact arrays and counters are shared across all
                    # worlds. Reset only fields that MJX itself vmaps. The next
                    # physics step rebuilds the shared collision workspace.
                    return current.replace(**{
                        field.name: select_reset(getattr(first, field.name), getattr(current, field.name))
                        for field in DataWarp.fields() if field.name not in DATA_NON_VMAP
                    })
                return select_reset(first, current)

            data = jax.tree_util.tree_map(
                reset_physics, cached["first_data"], stepped.data,
                is_leaf=lambda value: isinstance(value, DataWarp))
        else:
            data = jax.tree_util.tree_map(select_reset, cached["first_data"], stepped.data)
        return stepped.replace(
            data=data,
            obs=select_reset(cached["first_obs"], stepped.obs),
            done=done.astype(jnp.float32), info=info)
