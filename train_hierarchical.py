"""Train a reward-driven SAC manager and a Scaling-CRL worker in Brax."""

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

from envs.simple_maze import SimpleMaze
from hierarchical import HierarchicalAgent, LearnerConfig
from hierarchical_replay import ManagerReplay, WorkerReplay


@dataclass(frozen=True)
class Args(LearnerConfig):
    seed: int = 0
    backend: str = "generalized"
    target: tuple[float, float] = (12.0, 4.0)
    episode_length: int = 1000
    num_envs: int = 128
    num_eval_envs: int = 32
    total_env_steps: int = 100_000_000
    unroll_length: int = 64
    batch_size: int = 256
    max_replay_size: int = 10000  # primitive time steps per environment
    manager_replay_size: int = 2000  # completed intervals per environment
    min_replay_size: int = 1000  # primitive time steps per environment
    updates_per_collect: int = 64
    log_every: int = 10  # collection iterations
    eval_every: int = 100
    output_dir: str = ""
    resume: str = ""
    save_replay: bool = False
    capture_vis: bool = True
    vis_length: int = 1000
    track: bool = False
    wandb_project: str = "hierarchical-scaling-crl"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"


def make_env(args):
    return SimpleMaze(backend=args.backend, maze_layout_name="u_maze",
                      fixed_target=args.target, sparse_reward=True,
                      terminate_when_unhealthy=False)


@flax.struct.dataclass
class RolloutState:
    env_state: Any
    manager_actions: Any
    start_observations: Any
    duration: Any
    interval_return: Any
    episode_ids: Any

    @classmethod
    def create(cls, env_state):
        n = env_state.obs.shape[0]
        return cls(env_state, jnp.zeros((n, 2)), env_state.obs,
                   jnp.zeros(n, jnp.int32), jnp.zeros(n), jnp.zeros(n, jnp.int32))


def advance(agent, rollout, key, env, deterministic=False):
    """Shared commitment logic for collection, evaluation, and visualization."""
    manager_key, worker_key = jax.random.split(key)
    observations = rollout.env_state.obs
    decision = rollout.duration == 0
    proposed = agent.manager_actions(observations, manager_key, deterministic)
    manager_actions = jnp.where(decision[:, None], proposed, rollout.manager_actions)
    start_observations = jnp.where(decision[:, None], observations, rollout.start_observations)
    actions = agent.worker_actions(observations, agent.goals(manager_actions), worker_key, deterministic)
    next_state = env.step(rollout.env_state, actions)
    duration = rollout.duration + 1
    interval_return = rollout.interval_return + agent.config.manager_discount ** rollout.duration * next_state.reward
    done = next_state.done.astype(jnp.bool_)
    truncated = next_state.info["truncation"] > 0
    completed = (duration >= agent.config.subgoal_steps) | done
    manager_transition = {
        "observations": start_observations, "actions": manager_actions,
        "rewards": interval_return, "next_observations": next_state.obs,
        "duration": duration.astype(jnp.float32),
        "bootstrap": (~done).astype(jnp.float32),
        "valid": (~truncated).astype(jnp.float32),
    }
    next_rollout = rollout.replace(
        env_state=next_state, manager_actions=manager_actions,
        start_observations=start_observations,
        duration=jnp.where(completed, 0, duration),
        interval_return=jnp.where(completed, 0.0, interval_return),
        episode_ids=rollout.episode_ids + done.astype(jnp.int32),
    )
    return next_rollout, actions, manager_transition, completed, decision


def make_collector(env, unroll_length):
    @jax.jit
    def collect(agent, rollout, worker_replay, manager_replay, key):
        def step(carry, _):
            rollout, worker, manager, key = carry
            key, action_key = jax.random.split(key)
            next_rollout, actions, transition, completed, decision = advance(agent, rollout, action_key, env)
            worker = worker.insert(rollout.env_state.obs[:, :agent.state_dim], actions, rollout.episode_ids)
            manager = manager.insert(transition, completed)
            metrics = {
                "collect/reward": next_rollout.env_state.reward.mean(),
                "collect/success": next_rollout.env_state.metrics["success"].mean(),
                "collect/manager_decisions": decision.sum(),
                "collect/completed_intervals": completed.sum(),
                "collect/goal_saturation": (jnp.abs(next_rollout.manager_actions) > .99).mean(),
                "collect/masked_intervals": (completed * (1 - transition["valid"])).sum(),
            }
            return (next_rollout, worker, manager, key), metrics

        (rollout, worker_replay, manager_replay, key), metrics = jax.lax.scan(
            step, (rollout, worker_replay, manager_replay, key), None, length=unroll_length)
        return rollout, worker_replay, manager_replay, key, jax.tree_util.tree_map(jnp.mean, metrics)
    return collect


def make_learner(args):
    @jax.jit
    def learn(agent, worker_replay, manager_replay, key):
        def step(carry, _):
            agent, key = carry
            key, worker_key, manager_key, update_key = jax.random.split(key, 4)
            worker_batch = worker_replay.sample(worker_key, args.batch_size, args.worker_discount, args.episode_length)
            manager_batch = manager_replay.sample(manager_key, args.batch_size)
            agent, metrics = agent.update(worker_batch, manager_batch, update_key)
            return (agent, key), metrics
        (agent, key), metrics = jax.lax.scan(step, (agent, key), None, length=args.updates_per_collect)
        return agent, key, jax.tree_util.tree_map(jnp.mean, metrics)
    return learn


def make_evaluator(env, num_envs, episode_length):
    @jax.jit
    def evaluate(agent, key):
        key, reset_key = jax.random.split(key)
        rollout = RolloutState.create(env.reset(jax.random.split(reset_key, num_envs)))
        zeros = jnp.zeros(num_envs)

        def step(carry, _):
            rollout, key, active, returns, success, distance, lengths = carry
            key, action_key = jax.random.split(key)
            rollout, _, _, _, _ = advance(agent, rollout, action_key, env, deterministic=True)
            state = rollout.env_state
            returns += active * state.reward
            success = jnp.maximum(success, active * state.metrics["success"])
            distance = jnp.where(active, state.metrics["dist"], distance)
            lengths += active
            active = active & ~state.done.astype(jnp.bool_)
            return (rollout, key, active, returns, success, distance, lengths), None

        result, _ = jax.lax.scan(step, (rollout, key, jnp.ones(num_envs, bool), zeros, zeros, zeros, zeros), None, length=episode_length)
        _, _, _, returns, success, distance, lengths = result
        return {"eval/return": returns.mean(), "eval/success_rate": success.mean(),
                "eval/final_distance": distance.mean(), "eval/episode_length": lengths.mean()}
    return evaluate


def save_checkpoint(path, args, agent, key, env_steps, iteration, rollout, worker, manager):
    payload = {"config": asdict(args), "agent": flax.serialization.to_state_dict(agent),
               "key": key, "env_steps": env_steps, "iteration": iteration}
    if args.save_replay:
        payload["replay"] = {"worker": flax.serialization.to_state_dict(worker),
                             "manager": flax.serialization.to_state_dict(manager),
                             "rollout": flax.serialization.to_state_dict(rollout)}
    with Path(path).open("wb") as file:
        pickle.dump(jax.device_get(payload), file, protocol=pickle.HIGHEST_PROTOCOL)


def restore_agent(payload, agent):
    return flax.serialization.from_state_dict(agent, payload["agent"])


def render_policy(args, agent, key, output_dir):
    from brax.io import html

    base_env = make_env(args)
    env = wrappers.wrap(base_env, episode_length=args.episode_length)
    rollout = RolloutState.create(env.reset(jax.random.split(key, 1)))
    step = jax.jit(partial(advance, env=env, deterministic=True))
    states, commands = [], []
    for _ in range(args.vis_length):
        states.append(jax.tree_util.tree_map(lambda x: x[0], rollout.env_state.pipeline_state))
        key, step_key = jax.random.split(key)
        rollout, _, _, _, _ = step(agent, rollout, step_key)
        commands.append(np.asarray(agent.goals(rollout.manager_actions)[0]))
    (output_dir / "policy.html").write_text(html.render(base_env.sys, states))
    np.save(output_dir / "manager_goals.npy", np.asarray(commands))


def main(args):
    payload = None
    if args.resume:
        with Path(args.resume).open("rb") as file:
            payload = pickle.load(file)
        # Restore the experiment configuration; allow changing run/output controls.
        controls = {name: getattr(args, name) for name in (
            "total_env_steps", "output_dir", "resume", "save_replay", "capture_vis",
            "vis_length", "log_every", "eval_every", "track", "wandb_project", "wandb_entity", "wandb_mode")}
        args = replace(Args(**payload["config"]), **controls)
    if not (1 <= args.min_replay_size <= args.max_replay_size):
        raise ValueError("Require 1 <= min_replay_size <= max_replay_size.")
    if min(args.subgoal_steps, args.num_envs, args.num_eval_envs, args.unroll_length,
           args.batch_size, args.updates_per_collect, args.episode_length,
           args.manager_replay_size, args.log_every, args.eval_every) < 1:
        raise ValueError("Batch sizes, horizons, replay sizes, and intervals must be positive.")
    if not 0 < args.worker_discount <= 1 or not 0 < args.manager_discount <= 1:
        raise ValueError("Discounts must be in (0, 1].")
    output_dir = Path(args.output_dir or f"runs/hierarchical_{args.seed}_{datetime.now():%Y%m%d-%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(args), indent=2))
    run = None
    if args.track:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         mode=args.wandb_mode, config=asdict(args), dir=str(output_dir))

    config = LearnerConfig(**{field.name: getattr(args, field.name) for field in fields(LearnerConfig)})
    key, init_key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    env = wrappers.wrap(make_env(args), episode_length=args.episode_length)
    rollout = RolloutState.create(jax.jit(env.reset)(jax.random.split(reset_key, args.num_envs)))
    agent = HierarchicalAgent.create(init_key, config, observation_dim=rollout.env_state.obs.shape[-1])
    worker = WorkerReplay.create(args.max_replay_size, args.num_envs)
    manager = ManagerReplay.create(args.manager_replay_size, args.num_envs)
    env_steps, iteration = 0, 0
    if payload is not None:
        agent = restore_agent(payload, agent)
        key = jnp.asarray(payload["key"])
        env_steps, iteration = payload["env_steps"], payload["iteration"]
        if "replay" in payload:
            worker = flax.serialization.from_state_dict(worker, payload["replay"]["worker"])
            manager = flax.serialization.from_state_dict(manager, payload["replay"]["manager"])
            rollout = flax.serialization.from_state_dict(rollout, payload["replay"]["rollout"])

    collect = make_collector(env, args.unroll_length)
    learn = make_learner(args)
    evaluate = make_evaluator(env, args.num_eval_envs, args.episode_length)
    steps_per_collect = args.num_envs * args.unroll_length
    initial_steps, start_time = env_steps, time.monotonic()
    metrics = {}
    print(f"Training fixed-target point U-maze; output: {output_dir}", flush=True)

    def report(values):
        record = {name: float(np.asarray(value)) for name, value in values.items()}
        record.update(env_steps=env_steps, gradient_steps=int(agent.gradient_steps))
        with (output_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if run is not None:
            run.log(record, step=iteration)

    while env_steps < args.total_env_steps:
        rollout, worker, manager, key, metrics = collect(agent, rollout, worker, manager, key)
        env_steps += steps_per_collect
        iteration += 1
        ready = int(worker.size) >= args.min_replay_size and int(manager.sizes.sum()) > 0
        if ready:
            agent, key, learning_metrics = learn(agent, worker, manager, key)
            metrics = {**metrics, **learning_metrics}
        final = env_steps >= args.total_env_steps
        if iteration % args.log_every == 0 or iteration % args.eval_every == 0 or final:
            if ready:
                diagnostic_key = jax.random.fold_in(key, 1)
                batch = manager.sample(diagnostic_key, min(args.batch_size, 16))
                metrics.update(agent.gradient_diagnostics(batch["observations"], diagnostic_key))
            metrics.update({"replay/worker_steps_per_env": worker.size,
                            "replay/manager_intervals": manager.sizes.sum(),
                            "training/sps": (env_steps - initial_steps) / (time.monotonic() - start_time)})
            if iteration % args.eval_every == 0 or final:
                metrics.update(evaluate(agent, jax.random.fold_in(key, 2)))
                save_checkpoint(output_dir / "checkpoint.pkl", args, agent, key, env_steps,
                                iteration, rollout, worker, manager)
            report(metrics)

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
    import tyro
    main(tyro.cli(Args))
