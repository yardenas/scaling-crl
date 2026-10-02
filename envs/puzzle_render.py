"""Record and render one learned puzzle-policy episode."""

from dataclasses import replace
from functools import partial

import jax
import numpy as np
from brax.envs.wrappers import training as wrappers

from envs.puzzle_rollouts import PuzzleRolloutRecorder, render_rollout
from envs.reset_wrapper import FinalObservationWrapper


def render_policy(args, agent, key, output_dir):
    from train_hierarchical import RolloutState, advance, make_env

    raw_env = make_env(replace(args, num_eval_envs=1), evaluation=True)
    # Keep the real terminal pose and button pattern in the saved trajectory.
    env = FinalObservationWrapper(wrappers.VmapWrapper(
        wrappers.EpisodeWrapper(raw_env, args.episode_length, 1)))
    rollout = RolloutState.create(jax.jit(env.reset)(jax.random.split(key, 1)),
                                  agent.config.manager_action_dim)
    step = jax.jit(partial(advance, env=env, deterministic=True))
    recorder = PuzzleRolloutRecorder(rollout.env_state)
    commands = []
    for _ in range(args.vis_length):
        key, step_key = jax.random.split(key)
        rollout, actions, _, _, decision = step(agent, rollout, step_key)
        goals = (agent.goals(rollout.manager_actions) if args.manager_enabled else
                 rollout.env_state.obs[..., agent.state_dim:])
        recorder.append(rollout.env_state, actions, manager_goals=goals, manager_decisions=decision)
        commands.append(np.asarray(goals[0]))
        if bool(rollout.env_state.done[0]):
            break
    path = output_dir / "policy.npz"
    recorder.save(path, raw_env.simulator.mj_model, dict(
        env_name=raw_env.simulator._env_name, policy="learned", backend=args.backend,
        ctrl_dt=raw_env.simulator.dt, seed=args.seed))
    np.save(output_dir / "manager_goals.npy", np.asarray(commands))
    render_rollout(path)
