"""Standalone SAC-manager experiment with full physics-free puzzle episodes.

Run from the repository root:
    .venv/bin/python -m experiments.puzzle_logic.train
"""

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from experiments.puzzle_logic.puzzle_logic import PuzzleLogic
from hierarchical import HierarchicalAgent, LearnerConfig
from hierarchical_replay import ManagerReplay


@dataclass(frozen=True)
class Args:
    seed: int = 0
    task_id: int = 1
    num_envs: int = 128
    total_env_steps: int = 200_000
    episode_length: int = 50
    warmup_steps: int = 2_048
    replay_capacity: int = 65_536
    batch_size: int = 256
    # Gradient updates per vector step, which collects num_envs transitions.
    updates_per_step: int = 32
    manager_init_temperature: float = 0.01
    eval_interval: int = 10_000
    stochastic_eval_episodes: int = 100
    output_dir: str = ""


def learner_config(args=Args()):
    return LearnerConfig(
        freeze_worker=True, manager_worker_weight=0.0, subgoal_steps=1,
        manager_learn_duration=False, manager_count_bonus_scale=0.0,
        manager_action_candidates=1, goal_low=(0.0,) * 9, goal_high=(1.0,) * 9,
        manager_init_temperature=args.manager_init_temperature,
    )


def advance(env, state, lengths, actions, episode_length):
    """Execute one press per stream; reset only completed full episodes."""
    stepped = jax.vmap(env.step)(state, actions)
    lengths = lengths + 1
    terminal = stepped.done.astype(bool)
    timeout = lengths >= episode_length
    done = terminal | timeout
    transition = {
        "observations": state.obs, "actions": actions,
        "next_observations": stepped.obs, "rewards": stepped.reward,
        "duration": jnp.ones_like(stepped.reward),
        "bootstrap": (~terminal).astype(jnp.float32),
        "valid": jnp.ones_like(stepped.reward),
    }
    initial = env.reset(jax.random.PRNGKey(0))  # Fixed start; no reset randomness.

    def reset_leaf(current, first):
        mask = done.reshape(done.shape + (1,) * (current.ndim - 1))
        return jnp.where(mask, first, current)

    reset_state = jax.tree_util.tree_map(reset_leaf, stepped, initial)
    stats = {
        "episodes": done.sum(), "successes": terminal.sum(),
        "timeouts": (timeout & ~terminal).sum(),
        "episode_length_sum": jnp.where(done, lengths, 0).sum(),
    }
    return reset_state, jnp.where(done, 0, lengths), transition, stats


def make_train_step(env, args):
    @partial(jax.jit, static_argnames=("warmup",), donate_argnums=(1,))
    def step(agent, replay, state, lengths, key, warmup):
        key, action_key, sample_key, update_key = jax.random.split(key, 4)
        actions = (jax.random.uniform(action_key, (args.num_envs, 9), minval=-1, maxval=1)
                   if warmup else agent.manager_actions(state.obs, action_key))
        state, lengths, transition, stats = advance(
            env, state, lengths, actions, args.episode_length)
        replay = replay.insert(transition, jnp.ones(args.num_envs, dtype=bool))
        metrics = {}
        if not warmup:
            def update(carry, _):
                learner, sample_rng, update_rng = carry
                learner, losses = learner.update(
                    None, replay.sample(sample_rng, args.batch_size), update_rng)
                sample_rng = jax.random.split(sample_rng)[0]
                update_rng = jax.random.split(update_rng)[0]
                return (learner, sample_rng, update_rng), losses

            (agent, _, _), updates = jax.lax.scan(
                update, (agent, sample_key, update_key), None, length=args.updates_per_step)
            metrics = jax.tree_util.tree_map(lambda values: values.mean(axis=0), updates)
        return agent, replay, state, lengths, key, stats, metrics

    return step


def make_evaluator(env, args):
    """One deterministic rollout plus independent stochastic full episodes."""
    count = args.stochastic_eval_episodes + 1

    @jax.jit
    def evaluate(agent, key):
        key, reset_key = jax.random.split(key)
        state = jax.vmap(env.reset)(jax.random.split(reset_key, count))

        def step(carry, _):
            state, key, active = carry
            key, action_key = jax.random.split(key)
            actions = agent.manager_actions(state.obs, action_key)
            deterministic = agent.manager_actions(state.obs[:1], action_key, deterministic=True)
            actions = actions.at[:1].set(deterministic)
            stepped = jax.vmap(env.step)(state, actions)
            records = {
                "boards": stepped.obs[:, :9], "buttons": jnp.argmax(actions, axis=-1),
                "rewards": jnp.where(active, stepped.reward, 0), "active": active,
                "success": active & stepped.done.astype(bool),
            }
            return (stepped, key, active & ~stepped.done.astype(bool)), records

        _, records = jax.lax.scan(
            step, (state, key, jnp.ones(count, dtype=bool)), None, length=args.episode_length)
        return records

    return evaluate


def describe_evaluation(records, initial):
    records = jax.device_get(records)
    lengths = records["active"].sum(axis=0)
    successes = records["success"].any(axis=0)
    returns = records["rewards"].sum(axis=0)
    length = int(lengths[0])
    trajectory = {
        "success": bool(successes[0]), "presses": length, "return": float(returns[0]),
        "buttons_zero_based": records["buttons"][:length, 0].tolist(),
        "boards": [np.asarray(initial, dtype=int).tolist()]
        + records["boards"][:length, 0].astype(int).tolist(),
    }
    metrics = {
        "eval/deterministic_success": float(successes[0]),
        "eval/deterministic_presses": length,
        "eval/deterministic_return": float(returns[0]),
        "eval/stochastic_success_rate": float(successes[1:].mean()),
        "eval/stochastic_mean_presses": float(lengths[1:].mean()),
        "eval/stochastic_mean_return": float(returns[1:].mean()),
    }
    return metrics, trajectory


def shortest_solution(env):
    """Evaluation reference only; never used in collection, replay or updates."""
    initial, target = tuple(map(int, env.initial)), tuple(map(int, env.target))
    toggles = np.asarray(env.toggles)
    queue, visited = deque([(initial, [])]), {initial}
    while queue:
        board, buttons = queue.popleft()
        if board == target:
            return buttons
        for button, toggle in enumerate(toggles):
            following = tuple(np.bitwise_xor(board, toggle).tolist())
            if following not in visited:
                visited.add(following)
                queue.append((following, buttons + [button]))
    raise ValueError("Target cannot be reached from the initial board.")


def worker_digest(agent):
    worker = {name: getattr(agent, name) for name in
              ("worker_actor", "worker_critic", "worker_alpha")}
    # JIT can reorder dictionaries and convert Python counters to JAX scalars.
    # Hash canonical array leaves, not serialization's dictionary insertion order.
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(worker):
        array = np.asarray(jnp.asarray(leaf))
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(path, agent, args, env_steps):
    # Evaluation checkpoints: replay/rollout/RNG are deliberately not a resume format.
    payload = {"agent": flax.serialization.to_state_dict(agent),
               "args": asdict(args), "learner_config": asdict(agent.config),
               "env_steps": env_steps}
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(flax.serialization.to_bytes(payload))
    temporary.replace(path)


def main(args):
    for name in ("num_envs", "total_env_steps", "episode_length", "warmup_steps",
                 "replay_capacity", "batch_size", "updates_per_step", "eval_interval",
                 "stochastic_eval_episodes"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.replay_capacity % args.num_envs:
        raise ValueError("replay_capacity must be divisible by num_envs.")
    if not np.isfinite(args.manager_init_temperature) or args.manager_init_temperature <= 0:
        raise ValueError("manager_init_temperature must be finite and positive.")
    env = PuzzleLogic(task_id=args.task_id)
    output = Path(args.output_dir or
                  f"runs/puzzle_logic/{datetime.now():%Y%m%d-%H%M%S-%f}-seed{args.seed}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = learner_config(args)
    write_json(output / "config.json", {
        **asdict(args), "output_dir": str(output), "learner": asdict(config),
        "devices": [str(device) for device in jax.devices()], "jax_version": jax.__version__,
        "reward": "0 on success, -1 otherwise", "action_mapping": "argmax, row-major 0..8",
        "updates_per_training_transition": args.updates_per_step / args.num_envs,
        "checkpoint_contents": "agent and config only; not an exact training resume",
    })
    key, init_key, reset_key, eval_key = jax.random.split(jax.random.PRNGKey(args.seed), 4)
    agent = HierarchicalAgent.create(init_key, config, observation_dim=18, state_dim=9, action_dim=9)
    initial_worker_digest = worker_digest(agent)
    replay = ManagerReplay.create(args.replay_capacity // args.num_envs, args.num_envs,
                                  observation_dim=18, action_dim=9)
    state = jax.vmap(env.reset)(jax.random.split(reset_key, args.num_envs))
    lengths = jnp.zeros(args.num_envs, dtype=jnp.int32)
    train_step, evaluate = make_train_step(env, args), make_evaluator(env, args)
    reference = shortest_solution(env)
    write_json(output / "reference.json", {"optimal_presses": len(reference),
                                          "buttons_zero_based": reference})
    started = time.monotonic()
    best_rank, best_step, best_trajectory = None, 0, None
    totals = {"episodes": 0, "successes": 0, "timeouts": 0, "episode_length_sum": 0}
    latest_losses = {}

    def record(env_steps):
        nonlocal eval_key, best_rank, best_step, best_trajectory
        eval_key, episode_key = jax.random.split(eval_key)
        metrics, trajectory = describe_evaluation(evaluate(agent, episode_key), env.initial)
        row = {"env_steps": env_steps, "gradient_steps": int(agent.gradient_steps),
               "elapsed_seconds": time.monotonic() - started, **latest_losses, **metrics,
               **{f"train/{name}": value for name, value in totals.items()}}
        row["train/episode_success_rate"] = totals["successes"] / max(totals["episodes"], 1)
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / f"trajectory_{env_steps:09d}.json", trajectory)
        rank = (trajectory["success"], -trajectory["presses"], metrics["eval/stochastic_success_rate"])
        if best_rank is None or rank > best_rank:
            best_rank, best_step, best_trajectory = rank, env_steps, trajectory
            save_checkpoint(output / "best.msgpack", agent, args, env_steps)
        print(f"steps={env_steps} updates={int(agent.gradient_steps)} "
              f"det_success={trajectory['success']} presses={trajectory['presses']} "
              f"stochastic_success={metrics['eval/stochastic_success_rate']:.2%} "
              f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        return metrics, trajectory

    print(f"Output: {output}", flush=True)
    record(0)
    next_eval = args.eval_interval
    iterations = math.ceil(args.total_env_steps / args.num_envs)
    for iteration in range(iterations):
        warmup = iteration * args.num_envs < args.warmup_steps
        agent, replay, state, lengths, key, stats, losses = train_step(
            agent, replay, state, lengths, key, warmup=warmup)
        # Synchronize each vector step: fail promptly if learning becomes non-finite.
        host_stats, host_losses = jax.device_get((stats, losses))
        for name, value in host_stats.items():
            totals[name] += int(value)
        if host_losses:
            latest_losses = {name: float(value) for name, value in host_losses.items()}
            if not all(np.isfinite(value) for value in latest_losses.values()):
                raise FloatingPointError(f"Non-finite losses at vector step {iteration}: {latest_losses}")
        env_steps = (iteration + 1) * args.num_envs
        if env_steps >= next_eval or iteration == iterations - 1:
            final_metrics, final_trajectory = record(env_steps)
            next_eval = (env_steps // args.eval_interval + 1) * args.eval_interval

    unchanged = worker_digest(agent) == initial_worker_digest
    if not unchanged:
        raise AssertionError("Frozen worker changed during training.")
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(agent)):
        raise FloatingPointError("Non-finite final agent state.")
    save_checkpoint(output / "final.msgpack", agent, args, env_steps)
    summary = {
        "status": "solved" if final_trajectory["success"] else "not_solved",
        "env_steps": env_steps, "gradient_steps": int(agent.gradient_steps),
        "optimal_presses": len(reference), "final": final_trajectory,
        "best_step": best_step, "best": best_trajectory, "metrics": final_metrics,
        "worker_unchanged": unchanged, "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "result.json", summary)
    print(f"Result: {summary['status']}; artifacts: {output}", flush=True)
    return summary


if __name__ == "__main__":
    main(tyro.cli(Args))
