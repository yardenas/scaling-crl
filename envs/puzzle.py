"""Goal observations and batched episodic online rollouts for the MJX puzzle.

Layout: [achieved goal, simulator observation (55), commanded goal].
Goals are nine light bits, or end-effector XY plus nearest-button depression.
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

    def __init__(self, env_name, impl="warp", sparse=True, config_overrides=None, target=None,
                 random_board_goals=False, goal_mode="board", button_goal_depth=0.021,
                 manager_enabled=False):
        check_runtime(impl)
        from envs.ogbench_puzzle_mjx import OGBenchPuzzle3x3

        if goal_mode not in ("board", "button_xy_depression"):
            raise ValueError("puzzle_goal_mode must be board or button_xy_depression.")
        if goal_mode != "board" and random_board_goals:
            raise ValueError("Random board goals require puzzle_goal_mode=board.")
        self.goal_mode = goal_mode
        self.manager_enabled = manager_enabled
        self.goal_size = 9 if goal_mode == "board" else 3
        self.state_size = 55 + self.goal_size
        self.observation_size = self.state_size + self.goal_size
        overrides = dict(config_overrides or {})
        overrides.update(env_name=env_name, impl=impl, sparse=sparse,
                         target_button_states=target)
        if goal_mode == "button_xy_depression" and not manager_enabled:
            overrides["terminate_at_goal"] = False
        self.simulator = OGBenchPuzzle3x3(config_overrides=overrides)
        self.random_board_goals = random_board_goals
        if goal_mode == "button_xy_depression":
            import mujoco
            sim = self.simulator
            data = mujoco.MjData(sim.mj_model)
            data.qpos[:] = sim._init_qpos
            mujoco.mj_forward(sim.mj_model, data)
            self.button_xy = jnp.asarray(data.site_xpos[sim._button_site_ids, :2])
            joints = [sim.mj_model.joint(f"buttonbox_joint_{i}").id for i in range(9)]
            self.button_travel = jnp.asarray(-sim.mj_model.jnt_range[joints, 0])
            if not 0.02 < button_goal_depth <= float(self.button_travel.min()):
                raise ValueError("Button goal depth must exceed 0.02 m and fit within button travel.")
            self.button_goals = jnp.column_stack((self.button_xy, button_goal_depth / self.button_travel))

    @property
    def unwrapped(self):
        return self

    def _adapt(self, state):
        if self.goal_mode == "board":
            achieved = state.info["button_states"].astype(jnp.float32)
            target = state.info["target_button_states"].astype(jnp.float32)
            distance = jnp.sum(achieved != target).astype(jnp.float32)
        else:
            xy = state.data.site_xpos[self.simulator._pinch_site_id, :2]
            nearest = jnp.argmin(jnp.sum((self.button_xy - xy) ** 2, axis=-1))
            depression = jnp.clip(-state.data.qpos[self.simulator._button_qposadr] / self.button_travel, 0., 1.)
            achieved = jnp.concatenate((xy, depression[nearest][None]))
            target = state.info["worker_goal"]
            distance = (jnp.sum(state.info["button_states"] != state.info["target_button_states"])
                        if self.manager_enabled else jnp.linalg.norm(achieved - target))
        # A failed simulation terminates immediately. Keep its replay endpoint
        # finite so even a masked TD target cannot contaminate network gradients.
        obs = jnp.nan_to_num(jnp.concatenate((achieved, state.obs, target)),
                             nan=0.0, posinf=0.0, neginf=0.0)
        return state.replace(obs=obs, metrics={**state.metrics, "dist": distance})

    def reset(self, key, target=None):
        state = self.simulator.reset(key)
        if self.goal_mode == "board":
            if target is None and self.random_board_goals:
                target = jax.random.bernoulli(jax.random.fold_in(key, 1), shape=(9,)).astype(jnp.int32)
            if target is not None:
                state = state.replace(info={**state.info, "target_button_states": target.astype(jnp.int32)})
        else:
            if target is None:
                button = (0 if self.manager_enabled else
                          jax.random.randint(jax.random.fold_in(key, 1), (), 0, 9))
                target = self.button_goals[button]
            button = jnp.argmin(jnp.sum((self.button_xy - target[:2]) ** 2, axis=-1))
            state = state.replace(info={**state.info, "worker_goal": target, "goal_button": button})
            state = self._button_metrics(state, jnp.zeros(9, bool), jnp.asarray(False))
        return self._adapt(state)

    def step(self, state, action):
        stepped = self.simulator.step(state, action)
        if self.goal_mode == "button_xy_depression":
            before = state.data.qpos[self.simulator._button_qposadr]
            after = stepped.data.qpos[self.simulator._button_qposadr]
            presses = (before > -0.02) & (after <= -0.02)
            changed = jnp.any(stepped.info["button_states"] != state.info["button_states"])
            stepped = self._button_metrics(stepped, presses, changed)
            valid = stepped.metrics["valid"].astype(bool)
            success = stepped.metrics["success"].astype(bool)
            # Keep collecting after a press; only invalid physics or the wrapper's
            # time limit ends a button-goal rollout.
            if not self.manager_enabled:
                stepped = stepped.replace(reward=jnp.where(valid, success.astype(jnp.float32) - 1., stepped.reward))
        return self._adapt(stepped)

    def _button_metrics(self, state, presses, changed):
        if self.manager_enabled:
            # Task reward/success stay the native board objective. Press events
            # are diagnostics, including at the pre-autoreset endpoint.
            return state.replace(metrics={**state.metrics, "board_changed": changed.astype(jnp.float32),
                                           "button_presses": presses})
        button = state.info["goal_button"]
        success = presses[button] & state.metrics["valid"].astype(bool)
        return state.replace(info={**state.info, "success": success}, metrics={
            **state.metrics,
            "success": success.astype(jnp.float32), "board_changed": changed.astype(jnp.float32),
        })


class PuzzleTrainingEnv:
    """Vmap + time limits + complete cached autoreset, compatible with scan.

    Optional button-start training resamples the arm pose each episode.
    Both physical data and task info must reset: button bits aren't in mjx.Data.
    Preserve terminal reward/metrics and the pre-reset endpoint for SAC.
    """

    def __init__(self, env, episode_length, evaluation=False):
        self.env = env
        self.episode_length = episode_length
        self.action_size = env.action_size
        self.goal_size = env.goal_size
        self.evaluation = evaluation
        self.resample_resets = env.simulator._config.button_start_probability > 0

    @property
    def unwrapped(self):
        return self.env

    def reset(self, keys):
        # Use the same initial goal draws with and without the pose curriculum.
        keys = jax.vmap(jax.random.split)(keys)
        reset_keys = keys[:, 0]
        if self.evaluation and self.env.goal_mode == "button_xy_depression" and not self.env.manager_enabled:
            goals = self.env.button_goals[jnp.arange(reset_keys.shape[0]) % 9]
            state = jax.vmap(self.env.reset)(reset_keys, goals)
        else:
            state = jax.vmap(self.env.reset)(reset_keys)
        info = dict(state.info)
        if self.resample_resets:
            info["reset_rng"] = keys[:, 1]
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

        reset_data, reset_obs, reset_info = cached["first_data"], cached["first_obs"], cached["first_info"]
        if self.resample_resets:
            keys = jax.vmap(jax.random.split)(cached["reset_rng"])

            def reset(_):
                # Pose resampling must not change each world's assigned goal.
                fresh = jax.vmap(self.env.reset)(keys[:, 0], cached["first_obs"][:, -self.goal_size:])
                return fresh.data, fresh.obs, fresh.info

            reset_data, reset_obs, reset_info = jax.lax.cond(
                jnp.any(done), reset, lambda _: (stepped.data, stepped.obs, stepped.info), None)
        info = jax.tree_util.tree_map(select_reset, reset_info, stepped.info)
        if self.resample_resets:
            info["reset_rng"] = select_reset(keys[:, 1], cached["reset_rng"])
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
                reset_physics, reset_data, stepped.data,
                is_leaf=lambda value: isinstance(value, DataWarp))
        else:
            data = jax.tree_util.tree_map(select_reset, reset_data, stepped.data)
        return stepped.replace(
            data=data,
            obs=select_reset(reset_obs, stepped.obs),
            done=done.astype(jnp.float32), info=info)
