"""Train a reward-driven SAC manager and CRL worker in batched JAX environments."""

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import pickle
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as wrappers

from envs.reset_wrapper import FinalObservationWrapper
from hierarchical import HierarchicalAgent, LearnerConfig
from hierarchical_replay import ManagerReplay, WorkerReplay, relabel_manager_transition


@dataclass(frozen=True)
class Args(LearnerConfig):
    seed: int = 0
    env_id: str = "point_u_maze"
    eval_env_id: str | None = None
    backend: str = "generalized"
    env_config_overrides: dict[str, Any] | None = None
    # MJX-Warp naconmax is a global budget across the vmapped worlds.
    puzzle_contacts_per_env: int = 64
    puzzle_button_start_probability: float = 0.0
    puzzle_button_start_height: float = 0.04
    puzzle_random_board_goals: bool = False
    puzzle_goal_mode: str = "board"
    puzzle_button_goal_depth: float = 0.021
    target: tuple[float, ...] | None = (12.0, 4.0)
    terminate_on_success: bool = False
    manager_sparse_reward: bool = True
    manager_progress_reward: bool = False
    episode_length: int = 1000
    num_envs: int = 512
    num_eval_envs: int = 128
    total_env_steps: int = 100_000_000
    max_runtime_seconds: float | None = None
    unroll_length: int = 62
    batch_size: int = 512
    max_replay_size: int = 10000  # primitive time steps per environment
    manager_replay_size: int = 500  # completed intervals per environment
    min_replay_size: int = 1000  # primitive time steps per environment
    updates_per_collect: int = 800
    log_every: int = 10  # collection iterations
    eval_every: int = 100
    # Hydra uses environment-step intervals; None retains programmatic Args units.
    log_interval: int | None = None
    eval_interval: int | None = None
    save_interval: int | None = None
    run_group: str = "pointmaze-hierarchical"
    output_dir: str = ""
    resume: str = ""
    worker_checkpoint: str = ""
    save_replay: bool = False
    capture_vis: bool = True
    vis_length: int = 1000
    track: bool = False
    wandb_project: str = "hierarchical-scaling-crl"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"


def resolve_puzzle_goal_config(args):
    if args.env_id.replace("_", "-").startswith("puzzle-") and args.puzzle_goal_mode == "button_xy_depression":
        # Preserve the pretrained worker's continuous XY/depression interface.
        args = replace(args, goal_low=(0.25, -0.35, 0.), goal_high=(0.6, 0.35, 1.))
    return args


def make_env(args, evaluation=False):
    env_id = (args.eval_env_id or args.env_id) if evaluation else args.env_id
    if env_id.replace("_", "-").startswith("puzzle-"):
        from envs.puzzle import PuzzleEnv
        overrides = dict(args.env_config_overrides or {})
        batch_size = args.num_eval_envs if evaluation else args.num_envs
        overrides.setdefault("naconmax", batch_size * args.puzzle_contacts_per_env)
        overrides.setdefault("episode_length", args.episode_length)
        overrides["button_start_probability"] = 0.0 if evaluation else args.puzzle_button_start_probability
        overrides["button_start_height"] = args.puzzle_button_start_height
        if args.manager_enabled and args.puzzle_goal_mode == "button_xy_depression":
            overrides["terminate_at_goal"] = args.terminate_on_success
        return PuzzleEnv(env_id, impl=args.backend, sparse=args.manager_sparse_reward,
                         config_overrides=overrides, target=args.target,
                         random_board_goals=args.puzzle_random_board_goals and not evaluation,
                         goal_mode=args.puzzle_goal_mode, button_goal_depth=args.puzzle_button_goal_depth,
                         manager_enabled=args.manager_enabled)
    if env_id.startswith("ant_"):
        from envs.ant_maze import AntMaze
        return AntMaze(backend=args.backend, maze_layout_name=env_id[4:],
                       exclude_current_positions_from_observation=False,
                       terminate_when_unhealthy=True,
                       terminate_on_success=args.terminate_on_success,
                       sparse_reward=args.manager_enabled and args.manager_sparse_reward,
                       progress_reward=args.manager_enabled and args.manager_progress_reward,
                       fixed_target=args.target if args.manager_enabled else None)
    if env_id != "point_u_maze":
        raise ValueError(f"Unknown environment: {env_id}")
    from envs.simple_maze import SimpleMaze
    return SimpleMaze(backend=args.backend, maze_layout_name="u_maze",
                      fixed_target=args.target, sparse_reward=args.manager_sparse_reward,
                      terminate_when_unhealthy=False, terminate_on_success=args.terminate_on_success)


def wrap_env(args, evaluation=False):
    env = make_env(args, evaluation=evaluation)
    if hasattr(env, "simulator"):
        from envs.puzzle import PuzzleTrainingEnv
        return PuzzleTrainingEnv(env, args.episode_length, evaluation=evaluation)
    env = wrappers.EpisodeWrapper(env, episode_length=args.episode_length, action_repeat=1)
    env = FinalObservationWrapper(wrappers.VmapWrapper(env))
    return wrappers.AutoResetWrapper(env)


def commitment_steps(config, manager_actions):
    """Decode the held action in the rollout; SAC learns in tanh coordinates."""
    if not config.manager_learn_duration:
        return jnp.full(manager_actions.shape[:-1], config.subgoal_steps, jnp.int32)
    nominal = min(config.subgoal_steps, config.max_subgoal_steps)
    tau = manager_actions[..., config.goal_dim]
    log_scale = jnp.where(tau <= 0, tau * jnp.log(float(nominal)),
                          tau * jnp.log(config.max_subgoal_steps / nominal))
    return jnp.clip(jnp.rint(nominal * jnp.exp(log_scale)), 1, config.max_subgoal_steps).astype(jnp.int32)


@flax.struct.dataclass
class RolloutState:
    env_state: Any
    manager_actions: Any
    start_observations: Any
    duration: Any
    requested_steps: Any
    interval_return: Any
    episode_ids: Any
    counts: Any = None
    worker_goal_stats: Any = None

    @classmethod
    def create(cls, env_state, manager_action_dim=2, count_bins=0):
        n = env_state.obs.shape[0]
        return cls(env_state, jnp.zeros((n, manager_action_dim)), env_state.obs,
                   jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32),
                   jnp.zeros(n), jnp.zeros(n, jnp.int32),
                   jnp.zeros(count_bins ** 4, jnp.int32) if count_bins else None,
                   {name: jnp.zeros(n, bool) for name in ("reached", "initially_reached", "pressed", "wrong_press")}
                   if "worker_goal" in env_state.info else None)


def advance(agent, rollout, key, env, deterministic=False, worker_deterministic=None):
    """Shared commitment logic for collection, evaluation, and visualization."""
    manager_key, worker_key = jax.random.split(key)
    observations = rollout.env_state.obs
    worker_deterministic = deterministic if worker_deterministic is None else worker_deterministic
    if not agent.config.manager_enabled:
        goals = observations[..., agent.state_dim:agent.state_dim + agent.config.goal_dim]
        actions = agent.worker_actions(observations, goals, worker_key, worker_deterministic)
        next_state = env.step(rollout.env_state, actions)
        done = next_state.done.astype(jnp.int32)
        next_rollout = rollout.replace(env_state=next_state, episode_ids=rollout.episode_ids + done)
        inactive = jnp.zeros_like(done, dtype=jnp.bool_)
        return next_rollout, actions, None, inactive, inactive
    decision = rollout.duration == 0
    if not deterministic and agent.config.manager_action_candidates > 1:
        # Avoid the candidate critic evaluations when all environments are holding.
        proposed = jax.lax.cond(
            jnp.any(decision),
            lambda _: agent.exploratory_manager_actions(observations, manager_key, rollout.counts),
            lambda _: rollout.manager_actions, None)
    else:
        proposed = agent.manager_actions(observations, manager_key, deterministic)
    manager_actions = jnp.where(decision[:, None], proposed, rollout.manager_actions)
    counts = rollout.counts
    if counts is not None and not deterministic:
        # Scatter-add sums duplicate hits across environments. Never count held
        # commands, discarded candidates, or replay samples.
        indices = agent.count_indices(observations, manager_actions)
        counts = counts.at[indices].add(decision.astype(jnp.int32))
    requested_steps = jnp.where(decision, commitment_steps(agent.config, proposed), rollout.requested_steps)
    start_observations = jnp.where(decision[:, None], observations, rollout.start_observations)
    actions = agent.worker_actions(observations, agent.goals(manager_actions), worker_key, worker_deterministic)
    next_state = env.step(rollout.env_state, actions)
    duration = rollout.duration + 1
    reward_weight = (agent.config.manager_discount ** rollout.duration
                     if agent.config.manager_discount_per_step else 1.0)
    interval_return = rollout.interval_return + reward_weight * next_state.reward
    done = next_state.done.astype(jnp.bool_)
    truncated = next_state.info["truncation"] > 0
    completed = (duration >= requested_steps) | done
    final_available = next_state.info.get("final_observation_valid", jnp.zeros_like(done))
    endpoint = jnp.where((done & final_available)[:, None],
                         next_state.info.get("final_observation", next_state.obs), next_state.obs)
    terminal = done & ~truncated
    worker_goal_stats = rollout.worker_goal_stats
    if worker_goal_stats is not None:
        goals = agent.goals(manager_actions)
        valid = next_state.metrics["valid"].astype(bool)
        reached = worker_goal_match(endpoint[:, :3], goals) & valid
        button = jnp.argmin(jnp.sum((env.unwrapped.button_xy - goals[:, None, :2]) ** 2, axis=-1), axis=-1)
        presses = next_state.metrics["button_presses"] & valid[:, None]
        pressed = presses[jnp.arange(len(goals)), button]
        wrong_press = jnp.any(presses & (jnp.arange(9) != button[:, None]), axis=-1)
        worker_goal_stats = {
            "initially_reached": jnp.where(decision, worker_goal_match(observations[:, :3], goals),
                                           worker_goal_stats["initially_reached"]),
            **{name: jnp.where(decision, value, worker_goal_stats[name] | value)
               for name, value in (("reached", reached), ("pressed", pressed), ("wrong_press", wrong_press))},
        }
    manager_transition = {
        "observations": start_observations, "actions": manager_actions,
        "rewards": interval_return, "next_observations": endpoint,
        "duration": duration.astype(jnp.float32),
        "bootstrap": (~terminal).astype(jnp.float32),
        "valid": (~truncated | final_available).astype(jnp.float32),
    }
    next_rollout = rollout.replace(
        env_state=next_state, manager_actions=manager_actions,
        start_observations=start_observations,
        duration=jnp.where(completed, 0, duration),
        requested_steps=jnp.where(completed, 0, requested_steps),
        interval_return=jnp.where(completed, 0.0, interval_return),
        episode_ids=rollout.episode_ids + done.astype(jnp.int32),
        counts=counts,
        worker_goal_stats=worker_goal_stats,
    )
    return next_rollout, actions, manager_transition, completed, decision


def worker_goal_match(achieved, goal):
    """Continuous command attainment: 2 cm in XY, 0.1 normalized depression."""
    return ((jnp.linalg.norm(achieved[..., :2] - goal[..., :2], axis=-1) <= .02)
            & (jnp.abs(achieved[..., 2] - goal[..., 2]) <= .1))


def worker_command_statistics(env, agent, rollout, completed, active):
    if not agent.config.manager_enabled or rollout.worker_goal_stats is None:
        return {}
    goals = agent.goals(rollout.manager_actions)
    button = jnp.argmin(jnp.sum((env.unwrapped.button_xy - goals[:, None, :2]) ** 2, axis=-1), axis=-1)
    # Count physical press commands separately from shallow/release commands.
    press_command = ((jnp.linalg.norm(goals[:, :2] - env.unwrapped.button_xy[button], axis=-1) <= .02)
                     & (goals[:, 2] * env.unwrapped.button_travel[button] > .02))
    finished = completed & active
    stats = {name: (value & finished).sum() for name, value in rollout.worker_goal_stats.items()}
    return {**stats, "intervals": finished.sum(), "press_commands": (press_command & finished).sum(),
            "press_successes": (press_command & finished & rollout.worker_goal_stats["pressed"]).sum()}


def summarize_worker_commands(stats, prefix):
    if not stats:
        return {}
    totals = jax.tree_util.tree_map(jnp.sum, stats)
    n = jnp.maximum(totals["intervals"], 1)
    return {f"{prefix}/worker_goal_success_rate": totals["reached"] / n,
            f"{prefix}/worker_goal_initially_reached_fraction": totals["initially_reached"] / n,
            f"{prefix}/worker_wrong_button_fraction": totals["wrong_press"] / n,
            f"{prefix}/worker_press_command_fraction": totals["press_commands"] / n,
            f"{prefix}/worker_press_success_rate": totals["press_successes"] / jnp.maximum(totals["press_commands"], 1),
            f"{prefix}/worker_evaluated_intervals": totals["intervals"],
            f"{prefix}/worker_evaluated_press_commands": totals["press_commands"]}


def duration_statistics(config, actions, duration, decision, completed, active):
    """Unnormalized counts, so asynchronous decisions are weighted equally."""
    decision, completed = decision & active, completed & active
    requested = commitment_steps(config, actions)
    stats = {"decisions": decision.sum(), "completed": completed.sum(), "steps": active.sum(),
             "requested_sum": (requested * decision).sum(),
             "executed_sum": (duration * completed).sum()}
    lower = 0
    for upper in (1, 5, 25, 50, 100, 500):
        stats[f"requested_{lower + 1}_{upper}"] = (decision & (requested > lower) & (requested <= upper)).sum()
        lower = upper
    stats["requested_over_500"] = (decision & (requested > 500)).sum()
    if config.manager_learn_duration:
        tau = actions[..., config.goal_dim]
        stats.update(tau_sum=(tau * decision).sum(), tau_squared_sum=(tau ** 2 * decision).sum(),
                     tau_saturated=(decision & (jnp.abs(tau) > .99)).sum())
    return stats


def summarize_durations(stats, prefix):
    if not stats:
        return {}
    # Stats have a scan/time axis; each entry already sums over environments.
    totals = jax.tree_util.tree_map(jnp.sum, stats)
    decisions = jnp.maximum(totals["decisions"], 1)
    metrics = {f"{prefix}/manager_requested_steps": totals["requested_sum"] / decisions,
               f"{prefix}/manager_executed_steps": totals["executed_sum"] / jnp.maximum(totals["completed"], 1),
               f"{prefix}/manager_decision_fraction": totals["decisions"] / jnp.maximum(totals["steps"], 1)}
    metrics.update({f"{prefix}/manager_{name}_fraction": value / decisions
                    for name, value in totals.items() if name.startswith("requested_") and name != "requested_sum"})
    if "tau_sum" in totals:
        mean = totals["tau_sum"] / decisions
        metrics.update({f"{prefix}/manager_tau_mean": mean,
                        f"{prefix}/manager_tau_std": jnp.sqrt(jnp.maximum(totals["tau_squared_sum"] / decisions - mean ** 2, 0)),
                        f"{prefix}/manager_tau_saturation": totals["tau_saturated"] / decisions})
    return metrics


def make_collector(env, unroll_length):
    @jax.jit
    def collect(agent, rollout, worker_replay, manager_replay, key):
        def step(carry, _):
            rollout, worker, manager, key = carry
            key, action_key = jax.random.split(key)
            next_rollout, actions, transition, completed, decision = advance(agent, rollout, action_key, env)
            if not agent.config.freeze_worker:
                next_state = next_rollout.env_state
                endpoint = jnp.where(next_state.info["final_observation_valid"][:, None],
                                     next_state.info["final_observation"], next_state.obs)
                worker = worker.insert(rollout.env_state.obs[:, :agent.state_dim], actions, rollout.episode_ids,
                                       next_goals=endpoint[:, :agent.config.goal_dim])
            metrics = {
                "collect/reward": next_rollout.env_state.reward.mean(),
                "collect/success": next_rollout.env_state.metrics["success"].mean(),
            }
            for name in ("valid", "nan_termination", "ik_no_solution"):
                if name in next_rollout.env_state.metrics:
                    metrics[f"collect/{name}"] = next_rollout.env_state.metrics[name].mean()
            if "button_start" in rollout.env_state.info:
                near = rollout.env_state.info["button_start"]
                next_state = next_rollout.env_state
                done = next_state.done.astype(bool)
                endpoint = jnp.where(next_state.info["final_observation_valid"][:, None],
                                     next_state.info["final_observation"], next_state.obs)
                changed = (next_state.metrics["board_changed"].astype(bool) if "board_changed" in next_state.metrics else
                           jnp.any(endpoint[:, :agent.config.goal_dim] != rollout.env_state.obs[:, :agent.config.goal_dim], axis=-1))
                metrics.update({
                    "collect/button_start_step_fraction": near.mean(),
                    "collect/button_start_resets": (next_state.info["button_start"] & done).sum(),
                    "collect/completed_episodes": done.sum(),
                    "collect/button_change_fraction": changed.mean(),
                    "collect/button_start_change_fraction": (changed & near).sum() / jnp.maximum(near.sum(), 1),
                })
            duration_stats = {}
            if agent.config.manager_enabled:
                replay_transition = transition
                if agent.config.manager_hindsight_relabel:
                    replay_transition, relabel_metrics = relabel_manager_transition(
                        agent.config, transition, completed, next_rollout.env_state.done.astype(bool))
                    metrics.update(relabel_metrics)
                manager = manager.insert(replay_transition, completed)
                metrics.update({
                    "collect/manager_decisions": decision.sum(),
                    "collect/completed_intervals": completed.sum(),
                    "collect/goal_saturation": (jnp.abs(next_rollout.manager_actions[..., :agent.config.goal_dim]) > .99).mean(),
                    "collect/masked_intervals": (completed * (1 - transition["valid"])).sum(),
                })
                duration_stats = duration_statistics(agent.config, transition["actions"], transition["duration"],
                                                     decision, completed, jnp.ones_like(decision))
                if rollout.counts is not None:
                    bonus = agent.count_bonus(rollout.counts, rollout.env_state.obs, transition["actions"])
                    metrics["exploration/selection_bonus_sum"] = (bonus * decision).sum()
            worker_stats = worker_command_statistics(env, agent, next_rollout, completed,
                                                       jnp.ones_like(completed))
            return (next_rollout, worker, manager, key), (metrics, duration_stats, worker_stats)

        (rollout, worker_replay, manager_replay, key), (metrics, duration_stats, worker_stats) = jax.lax.scan(
            step, (rollout, worker_replay, manager_replay, key), None, length=unroll_length)
        metrics = {**jax.tree_util.tree_map(jnp.mean, metrics), **summarize_durations(duration_stats, "collect"),
                   **summarize_worker_commands(worker_stats, "collect")}
        if agent.config.manager_hindsight_relabel:
            metrics["relabel/fraction"] = metrics["relabel/relabeled_intervals"] / jnp.maximum(metrics["collect/completed_intervals"], 1e-8)
            metrics["relabel/mean_goal_shift"] = metrics.pop("relabel/goal_shift_sum") / jnp.maximum(metrics["relabel/relabeled_intervals"], 1e-8)
        if "collect/button_start_resets" in metrics:
            metrics["collect/button_start_reset_fraction"] = metrics["collect/button_start_resets"] / jnp.maximum(metrics["collect/completed_episodes"], 1e-8)
        if rollout.counts is not None:
            metrics["exploration/selection_bonus"] = metrics.pop("exploration/selection_bonus_sum") / jnp.maximum(metrics["collect/manager_decisions"], 1e-8)
            metrics.update({"exploration/visited_bins": (rollout.counts > 0).sum(),
                            "exploration/coverage": (rollout.counts > 0).mean(),
                            "exploration/total_commands": rollout.counts.sum(),
                            "exploration/max_count": rollout.counts.max()})
        return rollout, worker_replay, manager_replay, key, metrics
    return collect


def make_learner(args):
    @jax.jit
    def learn(agent, worker_replay, manager_replay, key, counts=None):
        def step(carry, _):
            agent, key = carry
            key, worker_key, manager_key, update_key = jax.random.split(key, 4)
            worker_batch = (None if args.freeze_worker else
                            worker_replay.sample(worker_key, args.batch_size, args.worker_discount, args.episode_length, args.goal_dim))
            manager_batch = manager_replay.sample(manager_key, args.batch_size) if args.manager_enabled else None
            bonus_metrics = {}
            if args.manager_enabled and args.manager_count_bonus_scale > 0:
                # Replay keeps task rewards; novelty decays with current counts.
                bonus = agent.count_bonus(counts, manager_batch["observations"], manager_batch["actions"])
                valid = manager_batch["valid"]
                mean = lambda x: (x * valid).sum() / jnp.maximum(valid.sum(), 1)
                bonus_metrics = {"exploration/replay_bonus": mean(bonus),
                                 "exploration/replay_task_reward": mean(manager_batch["rewards"])}
                manager_batch = {**manager_batch, "rewards": manager_batch["rewards"] + bonus}
            agent, metrics = agent.update(worker_batch, manager_batch, update_key)
            return (agent, key), {**metrics, **bonus_metrics}
        (agent, key), metrics = jax.lax.scan(step, (agent, key), None, length=args.updates_per_collect)
        return agent, key, jax.tree_util.tree_map(jnp.mean, metrics)
    return learn


def make_evaluator(env, num_envs, episode_length):
    @jax.jit
    def evaluate(agent, key):
        key, reset_key = jax.random.split(key)
        rollout = RolloutState.create(env.reset(jax.random.split(reset_key, num_envs)), agent.config.manager_action_dim)
        button_goals = (getattr(env.unwrapped, "goal_mode", "board") == "button_xy_depression"
                        and not agent.config.manager_enabled)
        # Normal puzzle resets are identical. Measure execution variability with
        # worker policy noise, retaining one fully deterministic reference episode.
        manager_buttons = (getattr(env.unwrapped, "goal_mode", "board") == "button_xy_depression"
                           and agent.config.manager_enabled)
        worker_deterministic = jnp.arange(num_envs) == 0 if manager_buttons else None
        if button_goals:
            commanded_buttons = rollout.env_state.info["goal_button"]
        zeros = jnp.zeros(num_envs)

        def step(carry, _):
            rollout, key, active, returns, success, success_steps, distance, lengths = carry
            key, action_key = jax.random.split(key)
            rollout, _, transition, completed, decision = advance(agent, rollout, action_key, env, deterministic=True,
                                                                   worker_deterministic=worker_deterministic)
            measured = active & (jnp.arange(num_envs) > 0) if manager_buttons and num_envs > 1 else active
            duration_stats = (duration_statistics(agent.config, transition["actions"], transition["duration"],
                                                   decision, completed, measured)
                              if agent.config.manager_enabled else {})
            worker_stats = worker_command_statistics(env, agent, rollout, completed, measured)
            state = rollout.env_state
            returns += active * state.reward
            success = jnp.maximum(success, active * state.metrics["success"])
            success_steps += active * state.metrics["success"]
            distance = jnp.where(active, state.metrics["dist"], distance)
            lengths += active
            active = active & ~state.done.astype(jnp.bool_)
            return (rollout, key, active, returns, success, success_steps, distance, lengths), (duration_stats, worker_stats)

        result, (duration_stats, worker_stats) = jax.lax.scan(step, (rollout, key, jnp.ones(num_envs, bool), zeros, zeros, zeros, zeros, zeros), None, length=episode_length)
        _, _, _, returns, success, success_steps, distance, lengths = result
        average = lambda x: x[1:].mean() if manager_buttons and num_envs > 1 else x.mean()
        metrics = {"eval/return": average(returns), "eval/success_rate": average(success),
                   "eval/success_steps": average(success_steps),
                   "eval/final_distance": average(distance), "eval/episode_length": average(lengths),
                   **summarize_durations(duration_stats, "eval"), **summarize_worker_commands(worker_stats, "eval")}
        if manager_buttons:
            metrics.update({"eval/deterministic_success": success[0],
                            "eval/stochastic_worker_episodes": jnp.asarray(num_envs - 1)})
        if button_goals:
            metrics["eval/button_success_rate"] = success.mean()
            for button in range(min(9, num_envs)):
                mask = commanded_buttons == button
                metrics[f"eval/button_{button}_success_rate"] = jnp.sum(success * mask) / mask.sum()
        return metrics
    return evaluate


def save_checkpoint(path, args, agent, key, env_steps, iteration, rollout, worker, manager):
    payload = {"config": asdict(args), "agent": flax.serialization.to_state_dict(agent),
               "key": key, "env_steps": env_steps, "iteration": iteration,
               "rollout": flax.serialization.to_state_dict(rollout)}
    if args.save_replay:
        payload["replay"] = {"worker": flax.serialization.to_state_dict(worker),
                             "manager": flax.serialization.to_state_dict(manager)}
    with Path(path).open("wb") as file:
        pickle.dump(jax.device_get(payload), file, protocol=pickle.HIGHEST_PROTOCOL)


def restore_agent(payload, agent):
    return flax.serialization.from_state_dict(agent, payload["agent"])


WORKER_STATES = ("worker_actor", "worker_critic", "worker_alpha")
WORKER_ARCHITECTURE = ("actor_network_width", "critic_network_width", "actor_depth", "critic_depth", "use_relu")


def restore_worker(worker_state, agent):
    restored = {}
    for name in WORKER_STATES:
        template = getattr(agent, name)
        state = flax.serialization.from_state_dict(template, worker_state[name])
        shapes = lambda params: [x.shape for x in jax.tree_util.tree_leaves(params)]
        if shapes(state.params) != shapes(template.params):
            raise ValueError("Worker checkpoint has incompatible state/action dimensions for this environment.")
        restored[name] = state
    return agent.replace(**restored)


def render_policy(args, agent, key, output_dir):
    if args.env_id.replace("_", "-").startswith("puzzle-"):
        from envs.puzzle_render import render_policy as render_puzzle
        return render_puzzle(args, agent, key, output_dir)
    from brax.io import html

    env = wrap_env(args, evaluation=True)
    rollout = RolloutState.create(env.reset(jax.random.split(key, 1)), agent.config.manager_action_dim)
    step = jax.jit(partial(advance, env=env, deterministic=True))
    states, commands, durations, decisions = [], [], [], []
    for _ in range(args.vis_length):
        states.append(jax.tree_util.tree_map(lambda x: x[0], rollout.env_state.pipeline_state))
        key, step_key = jax.random.split(key)
        rollout, _, _, _, decision = step(agent, rollout, step_key)
        goals = (agent.goals(rollout.manager_actions) if args.manager_enabled
                 else rollout.env_state.obs[..., agent.state_dim:agent.state_dim + agent.config.goal_dim])
        commands.append(np.asarray(goals[0]))
        durations.append(np.asarray(commitment_steps(agent.config, rollout.manager_actions)[0]))
        decisions.append(np.asarray(decision[0]))
    (output_dir / "policy.html").write_text(html.render(env.unwrapped.sys, states))
    np.save(output_dir / "manager_goals.npy", np.asarray(commands))
    np.save(output_dir / "manager_requested_steps.npy", np.asarray(durations))
    np.save(output_dir / "manager_decisions.npy", np.asarray(decisions))


def main(args, tracking_config=None):
    run_start = time.monotonic()
    payload = None
    worker_state = None
    if args.resume and args.worker_checkpoint:
        raise ValueError("Use resume for continuing a run, or worker_checkpoint for a fresh manager.")
    if args.resume:
        with Path(args.resume).open("rb") as file:
            payload = pickle.load(file)
        # Restore the experiment configuration; allow changing run/output controls.
        controls = {name: getattr(args, name) for name in (
            "total_env_steps", "max_runtime_seconds", "output_dir", "resume", "save_replay", "capture_vis",
            "vis_length", "log_every", "eval_every", "log_interval", "eval_interval",
            "save_interval", "run_group", "track", "wandb_project", "wandb_entity", "wandb_mode")}
        saved_config = dict(payload["config"])
        # Earlier puzzle checkpoints stored an unused two-coordinate maze target.
        if saved_config["env_id"].replace("_", "-").startswith("puzzle-") and len(saved_config.get("target") or ()) == 2:
            saved_config["target"] = None
        args = replace(Args(**saved_config), **controls)
    elif args.worker_checkpoint:
        with Path(args.worker_checkpoint).open("rb") as file:
            pretrained = pickle.load(file)
        args = replace(args, **{name: pretrained["config"][name] for name in WORKER_ARCHITECTURE})
        worker_state = {name: pretrained["agent"][name] for name in WORKER_STATES}
        del pretrained  # Do not keep the source replay in memory.
    args = resolve_puzzle_goal_config(args)
    if args.freeze_worker and not args.manager_enabled:
        raise ValueError("freeze_worker requires manager_enabled=true.")
    if args.freeze_worker and not (args.resume or args.worker_checkpoint):
        raise ValueError("freeze_worker requires a pretrained worker_checkpoint (or resume).")
    if not 0 <= args.puzzle_button_start_probability <= 1 or args.puzzle_button_start_height <= 0:
        raise ValueError("Require puzzle_button_start_probability in [0, 1] and positive height.")
    if not (1 <= args.min_replay_size <= args.max_replay_size):
        raise ValueError("Require 1 <= min_replay_size <= max_replay_size.")
    if min(args.subgoal_steps, args.max_subgoal_steps, args.num_envs, args.num_eval_envs, args.unroll_length,
           args.batch_size, args.updates_per_collect, args.episode_length,
           args.manager_replay_size, args.log_every, args.eval_every) < 1:
        raise ValueError("Batch sizes, horizons, replay sizes, and intervals must be positive.")
    if not 0 < args.worker_discount <= 1 or not 0 < args.manager_discount <= 1:
        raise ValueError("Discounts must be in (0, 1].")
    if args.manager_count_bins < 1 or args.manager_action_candidates < 1 or args.manager_count_bonus_scale < 0:
        raise ValueError("Count bins/candidates must be positive and bonus scale nonnegative.")
    if not args.goal_low or len(args.goal_low) != len(args.goal_high) or any(
            low >= high for low, high in zip(args.goal_low, args.goal_high)):
        raise ValueError("Goal bounds must have matching dimensions with low < high.")
    if args.goal_dim != 2 and args.manager_count_bonus_scale > 0:
        raise ValueError("The XY count bonus is only supported for two-dimensional maze goals.")
    if args.manager_hindsight_relabel:
        if not args.manager_enabled or args.goal_dim != 2 or not (args.env_id.startswith("ant_") or args.env_id == "point_u_maze"):
            raise ValueError("Manager hindsight relabeling currently supports two-dimensional maze goals only.")
        if not np.isfinite(args.manager_hindsight_goal_tolerance) or args.manager_hindsight_goal_tolerance <= 0:
            raise ValueError("Manager hindsight goal tolerance must be finite and positive.")
        if args.manager_count_bonus_scale != 0:
            raise ValueError("Manager hindsight pilot requires count bonuses disabled; original-command counts do not track relabeled commands.")
    output_dir = Path(args.output_dir or f"runs/hierarchical_{args.seed}_{datetime.now():%Y%m%d-%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(args), indent=2))
    run = None
    if args.track:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.run_group, name=output_dir.name,
                         mode=args.wandb_mode,
                         config={**(tracking_config or {}), "training": asdict(args)},
                         dir=str(output_dir))

    config = LearnerConfig(**{field.name: getattr(args, field.name) for field in fields(LearnerConfig)})
    key, init_key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    env = wrap_env(args)
    if getattr(env.unwrapped, "goal_size", 2) != args.goal_dim:
        raise ValueError("Goal bounds must match the environment's achieved-goal dimensions.")
    rollout = RolloutState.create(jax.jit(env.reset)(jax.random.split(reset_key, args.num_envs)), config.manager_action_dim,
                                  config.manager_count_bins if config.manager_enabled and config.manager_count_bonus_scale > 0 else 0)
    observation_dim = rollout.env_state.obs.shape[-1]
    state_dim, action_dim = observation_dim - config.goal_dim, env.action_size
    agent = HierarchicalAgent.create(init_key, config, observation_dim=observation_dim,
                                     state_dim=state_dim, action_dim=action_dim)
    if worker_state is not None:
        agent = restore_worker(worker_state, agent)
        del worker_state
        print(f"Loaded worker from {args.worker_checkpoint}; freeze_worker={args.freeze_worker}. Manager starts fresh.", flush=True)
    worker = WorkerReplay.create(1 if args.freeze_worker else args.max_replay_size,
                                 args.num_envs, state_dim, action_dim,
                                 goal_dim=config.goal_dim if hasattr(env.unwrapped, "simulator") else None)
    manager = ManagerReplay.create(args.manager_replay_size if args.manager_enabled else 1,
                                   args.num_envs, observation_dim=observation_dim, action_dim=config.manager_action_dim)
    env_steps, iteration = 0, 0
    if payload is not None:
        agent = restore_agent(payload, agent)
        key = jnp.asarray(payload["key"])
        env_steps, iteration = payload["env_steps"], payload["iteration"]
        rollout = flax.serialization.from_state_dict(rollout, payload["rollout"])
        if "replay" in payload:
            worker_payload = dict(payload["replay"]["worker"])
            # Older maze checkpoints predate optional endpoint-goal storage.
            worker_payload.setdefault("next_goals", None)
            worker = flax.serialization.from_state_dict(worker, worker_payload)
            manager = flax.serialization.from_state_dict(manager, payload["replay"]["manager"])

    collect = make_collector(env, args.unroll_length)
    learn = make_learner(args)
    eval_env = (wrap_env(args, evaluation=True)
                if hasattr(env.unwrapped, "simulator") or (args.eval_env_id and args.eval_env_id != args.env_id) else env)
    evaluate = make_evaluator(eval_env, args.num_eval_envs, args.episode_length)
    worker_evaluate = None
    if args.manager_enabled and args.puzzle_goal_mode == "button_xy_depression":
        probe_args = replace(args, manager_enabled=False, freeze_worker=False, num_eval_envs=9)
        worker_evaluate = make_evaluator(wrap_env(probe_args, evaluation=True), 9, args.episode_length)
    best_evaluation = (-float("inf"), -float("inf"))
    steps_per_collect = args.num_envs * args.unroll_length
    initial_steps, start_time = env_steps, time.monotonic()
    warmup_steps = args.min_replay_size * args.num_envs
    if payload is not None and "replay" in payload:
        warmup_steps = max(0, warmup_steps - env_steps)
    del payload
    metrics = {}
    print(f"Training {args.env_id}; state_dim={state_dim}, action_dim={action_dim}; output: {output_dir}", flush=True)

    def due(interval, previous_steps):
        return interval > 0 and env_steps // interval > previous_steps // interval

    def report(values):
        record = {name: float(np.asarray(value)) for name, value in values.items()}
        record.update(env_steps=env_steps, gradient_steps=int(agent.gradient_steps))
        with (output_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if run is not None:
            run.log(record, step=iteration)

    while env_steps < args.total_env_steps:
        previous_steps = env_steps
        rollout, worker, manager, key, metrics = collect(agent, rollout, worker, manager, key)
        env_steps += steps_per_collect
        iteration += 1
        ready = (env_steps - initial_steps >= warmup_steps if args.freeze_worker else
                 int(worker.size) >= args.min_replay_size)
        ready = ready and (not args.manager_enabled or int(manager.sizes.sum()) > 0)
        if ready:
            agent, key, learning_metrics = learn(agent, worker, manager, key, rollout.counts)
            metrics = {**metrics, **learning_metrics}
        time_limit_reached = (args.max_runtime_seconds is not None
                              and time.monotonic() - run_start >= args.max_runtime_seconds)
        final = env_steps >= args.total_env_steps or time_limit_reached
        log_due = due(args.log_interval, previous_steps) if args.log_interval is not None else iteration % args.log_every == 0
        eval_due = due(args.eval_interval, previous_steps) if args.eval_interval is not None else iteration % args.eval_every == 0
        save_due = due(args.save_interval, previous_steps) if args.save_interval is not None else eval_due
        if log_due or eval_due or final:
            if ready and args.manager_enabled:
                diagnostic_key = jax.random.fold_in(key, 1)
                batch = manager.sample(diagnostic_key, min(args.batch_size, 16))
                metrics.update(agent.gradient_diagnostics(batch["observations"], diagnostic_key))
            metrics.update({"replay/worker_steps_per_env": worker.size,
                            "replay/manager_intervals": manager.sizes.sum(),
                            "training/sps": (env_steps - initial_steps) / (time.monotonic() - start_time)})
            if eval_due or final:
                evaluation = evaluate(agent, jax.random.fold_in(key, 2))
                metrics.update(evaluation)
                if worker_evaluate is not None:
                    probe = agent.replace(config=replace(agent.config, manager_enabled=False))
                    metrics.update({name.replace("eval/", "worker_eval/"): value for name, value in
                                    worker_evaluate(probe, jax.random.PRNGKey(2026)).items()})
                selection = (float(evaluation["eval/success_rate"]), float(evaluation["eval/return"]))
                if selection > best_evaluation:
                    best_evaluation = selection
                    save_checkpoint(output_dir / "best_checkpoint.pkl", replace(args, save_replay=False),
                                    agent, key, env_steps, iteration, rollout, worker, manager)
            report(metrics)
        if save_due or final:
            save_checkpoint(output_dir / "checkpoint.pkl", args, agent, key, env_steps,
                            iteration, rollout, worker, manager)
        if time_limit_reached:
            print("Runtime budget reached; saved final evaluation and checkpoint.", flush=True)
            break

    # Also save/evaluate a resumed run whose requested budget is already complete.
    if env_steps == initial_steps:
        report(evaluate(agent, jax.random.fold_in(key, 2)))
        save_checkpoint(output_dir / "checkpoint.pkl", args, agent, key, env_steps,
                        iteration, rollout, worker, manager)
    if args.capture_vis:
        render_policy(args, agent, jax.random.fold_in(key, 3), output_dir)
    if run is not None:
        run.finish()
    return agent


if __name__ == "__main__":
    from main import main as hydra_main
    hydra_main()
