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
from envs.reset_wrapper import FinalObservationWrapper, ResamplingAutoResetWrapper
from hierarchical import HierarchicalAgent, LearnerConfig
from hierarchical_replay import ManagerReplay, WorkerReplay


@dataclass(frozen=True)
class Args(LearnerConfig):
    seed: int = 0
    env_id: str = "point_u_maze"
    eval_env_id: str | None = None
    backend: str = "generalized"
    target: tuple[float, float] = (12.0, 4.0)
    manager_sparse_reward: bool = True
    manager_progress_reward: bool = False
    goal_start_probability: float = 0.0
    goal_start_radius: float = 0.25
    episode_length: int = 1000
    num_envs: int = 128
    num_eval_envs: int = 32
    total_env_steps: int = 100_000_000
    max_runtime_seconds: float | None = None
    unroll_length: int = 64
    batch_size: int = 256
    max_replay_size: int = 10000  # primitive time steps per environment
    manager_replay_size: int = 2000  # completed intervals per environment
    min_replay_size: int = 1000  # primitive time steps per environment
    updates_per_collect: int = 64
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


def make_env(args, evaluation=False):
    env_id = (args.eval_env_id or args.env_id) if evaluation else args.env_id
    if env_id.startswith("ant_"):
        from envs.ant_maze import AntMaze
        return AntMaze(backend=args.backend, maze_layout_name=env_id[4:],
                       exclude_current_positions_from_observation=False,
                       terminate_when_unhealthy=True,
                       sparse_reward=args.manager_enabled and args.manager_sparse_reward,
                       progress_reward=args.manager_enabled and args.manager_progress_reward,
                       fixed_target=args.target if args.manager_enabled else None,
                       goal_start_probability=0.0 if evaluation else args.goal_start_probability,
                       goal_start_radius=args.goal_start_radius)
    if env_id != "point_u_maze":
        raise ValueError(f"Unknown environment: {env_id}")
    return SimpleMaze(backend=args.backend, maze_layout_name="u_maze",
                      fixed_target=args.target, sparse_reward=args.manager_sparse_reward,
                      terminate_when_unhealthy=False)


def wrap_env(args, evaluation=False):
    env = make_env(args, evaluation=evaluation)
    env = wrappers.EpisodeWrapper(env, episode_length=args.episode_length, action_repeat=1)
    env = FinalObservationWrapper(wrappers.VmapWrapper(env))
    if args.goal_start_probability > 0 and not evaluation:
        return ResamplingAutoResetWrapper(env)
    return wrappers.AutoResetWrapper(env)


def commitment_steps(config, manager_actions):
    """Decode the held action in the rollout; SAC learns in tanh coordinates."""
    if not config.manager_learn_duration:
        return jnp.full(manager_actions.shape[:-1], config.subgoal_steps, jnp.int32)
    nominal = min(config.subgoal_steps, config.max_subgoal_steps)
    tau = manager_actions[..., 2]
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

    @classmethod
    def create(cls, env_state, manager_action_dim=2):
        n = env_state.obs.shape[0]
        return cls(env_state, jnp.zeros((n, manager_action_dim)), env_state.obs,
                   jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32),
                   jnp.zeros(n), jnp.zeros(n, jnp.int32))


def advance(agent, rollout, key, env, deterministic=False):
    """Shared commitment logic for collection, evaluation, and visualization."""
    manager_key, worker_key = jax.random.split(key)
    observations = rollout.env_state.obs
    if not agent.config.manager_enabled:
        goals = observations[..., agent.state_dim:agent.state_dim + 2]
        actions = agent.worker_actions(observations, goals, worker_key, deterministic)
        next_state = env.step(rollout.env_state, actions)
        done = next_state.done.astype(jnp.int32)
        next_rollout = rollout.replace(env_state=next_state, episode_ids=rollout.episode_ids + done)
        inactive = jnp.zeros_like(done, dtype=jnp.bool_)
        return next_rollout, actions, None, inactive, inactive
    decision = rollout.duration == 0
    proposed = agent.manager_actions(observations, manager_key, deterministic)
    manager_actions = jnp.where(decision[:, None], proposed, rollout.manager_actions)
    requested_steps = jnp.where(decision, commitment_steps(agent.config, proposed), rollout.requested_steps)
    start_observations = jnp.where(decision[:, None], observations, rollout.start_observations)
    actions = agent.worker_actions(observations, agent.goals(manager_actions), worker_key, deterministic)
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
    )
    return next_rollout, actions, manager_transition, completed, decision


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
        tau = actions[..., 2]
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
                worker = worker.insert(rollout.env_state.obs[:, :agent.state_dim], actions, rollout.episode_ids)
            metrics = {
                "collect/reward": next_rollout.env_state.reward.mean(),
                "collect/success": next_rollout.env_state.metrics["success"].mean(),
            }
            duration_stats = {}
            if "goal_start" in rollout.env_state.info:
                near = rollout.env_state.info["goal_start"]
                done = next_rollout.env_state.done
                reward = next_rollout.env_state.reward
                metrics.update({
                    "collect/goal_start_fraction": near.mean(),
                    "collect/goal_start_episodes": (near * done).sum(),
                    "collect/completed_episodes": done.sum(),
                    "collect/goal_start_reward": jnp.sum(near * reward) / jnp.maximum(near.sum(), 1),
                    "collect/normal_start_reward": jnp.sum(~near * reward) / jnp.maximum((~near).sum(), 1),
                })
            if agent.config.manager_enabled:
                manager = manager.insert(transition, completed)
                metrics.update({
                    "collect/manager_decisions": decision.sum(),
                    "collect/completed_intervals": completed.sum(),
                    "collect/goal_saturation": (jnp.abs(next_rollout.manager_actions[..., :2]) > .99).mean(),
                    "collect/masked_intervals": (completed * (1 - transition["valid"])).sum(),
                })
                duration_stats = duration_statistics(agent.config, transition["actions"], transition["duration"],
                                                     decision, completed, jnp.ones_like(decision))
            return (next_rollout, worker, manager, key), (metrics, duration_stats)

        (rollout, worker_replay, manager_replay, key), (metrics, duration_stats) = jax.lax.scan(
            step, (rollout, worker_replay, manager_replay, key), None, length=unroll_length)
        metrics = {**jax.tree_util.tree_map(jnp.mean, metrics), **summarize_durations(duration_stats, "collect")}
        return rollout, worker_replay, manager_replay, key, metrics
    return collect


def make_learner(args):
    @jax.jit
    def learn(agent, worker_replay, manager_replay, key):
        def step(carry, _):
            agent, key = carry
            key, worker_key, manager_key, update_key = jax.random.split(key, 4)
            worker_batch = (None if args.freeze_worker else
                            worker_replay.sample(worker_key, args.batch_size, args.worker_discount, args.episode_length))
            manager_batch = manager_replay.sample(manager_key, args.batch_size) if args.manager_enabled else None
            agent, metrics = agent.update(worker_batch, manager_batch, update_key)
            return (agent, key), metrics
        (agent, key), metrics = jax.lax.scan(step, (agent, key), None, length=args.updates_per_collect)
        return agent, key, jax.tree_util.tree_map(jnp.mean, metrics)
    return learn


def make_evaluator(env, num_envs, episode_length):
    @jax.jit
    def evaluate(agent, key):
        key, reset_key = jax.random.split(key)
        rollout = RolloutState.create(env.reset(jax.random.split(reset_key, num_envs)), agent.config.manager_action_dim)
        zeros = jnp.zeros(num_envs)

        def step(carry, _):
            rollout, key, active, returns, success, success_steps, distance, lengths = carry
            key, action_key = jax.random.split(key)
            rollout, _, transition, completed, decision = advance(agent, rollout, action_key, env, deterministic=True)
            duration_stats = (duration_statistics(agent.config, transition["actions"], transition["duration"],
                                                   decision, completed, active)
                              if agent.config.manager_enabled else {})
            state = rollout.env_state
            returns += active * state.reward
            success = jnp.maximum(success, active * state.metrics["success"])
            success_steps += active * state.metrics["success"]
            distance = jnp.where(active, state.metrics["dist"], distance)
            lengths += active
            active = active & ~state.done.astype(jnp.bool_)
            return (rollout, key, active, returns, success, success_steps, distance, lengths), duration_stats

        result, duration_stats = jax.lax.scan(step, (rollout, key, jnp.ones(num_envs, bool), zeros, zeros, zeros, zeros, zeros), None, length=episode_length)
        _, _, _, returns, success, success_steps, distance, lengths = result
        return {"eval/return": returns.mean(), "eval/success_rate": success.mean(),
                "eval/success_steps": success_steps.mean(),
                "eval/final_distance": distance.mean(), "eval/episode_length": lengths.mean(),
                **summarize_durations(duration_stats, "eval")}
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
                 else rollout.env_state.obs[..., agent.state_dim:agent.state_dim + 2])
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
        args = replace(Args(**payload["config"]), **controls)
    elif args.worker_checkpoint:
        with Path(args.worker_checkpoint).open("rb") as file:
            pretrained = pickle.load(file)
        args = replace(args, **{name: pretrained["config"][name] for name in WORKER_ARCHITECTURE})
        worker_state = {name: pretrained["agent"][name] for name in WORKER_STATES}
        del pretrained  # Do not keep the source replay in memory.
    if args.freeze_worker and not args.manager_enabled:
        raise ValueError("freeze_worker requires manager_enabled=true.")
    if args.freeze_worker and not (args.resume or args.worker_checkpoint):
        raise ValueError("freeze_worker requires a pretrained worker_checkpoint (or resume).")
    if not 0 <= args.goal_start_probability <= 1 or not 0 <= args.goal_start_radius <= 0.5:
        raise ValueError("Require goal_start_probability in [0, 1] and goal_start_radius in [0, 0.5].")
    if args.goal_start_probability > 0 and not args.env_id.startswith("ant_"):
        raise ValueError("Goal-start training is currently supported for AntMaze.")
    if not (1 <= args.min_replay_size <= args.max_replay_size):
        raise ValueError("Require 1 <= min_replay_size <= max_replay_size.")
    if min(args.subgoal_steps, args.max_subgoal_steps, args.num_envs, args.num_eval_envs, args.unroll_length,
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
                         group=args.run_group, name=output_dir.name,
                         mode=args.wandb_mode,
                         config={**(tracking_config or {}), "training": asdict(args)},
                         dir=str(output_dir))

    config = LearnerConfig(**{field.name: getattr(args, field.name) for field in fields(LearnerConfig)})
    key, init_key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    env = wrap_env(args)
    rollout = RolloutState.create(jax.jit(env.reset)(jax.random.split(reset_key, args.num_envs)), config.manager_action_dim)
    observation_dim = rollout.env_state.obs.shape[-1]
    state_dim, action_dim = observation_dim - 2, env.action_size
    agent = HierarchicalAgent.create(init_key, config, observation_dim=observation_dim,
                                     state_dim=state_dim, action_dim=action_dim)
    if worker_state is not None:
        agent = restore_worker(worker_state, agent)
        del worker_state
        print(f"Loaded worker from {args.worker_checkpoint}; freeze_worker={args.freeze_worker}. Manager starts fresh.", flush=True)
    worker = WorkerReplay.create(1 if args.freeze_worker else args.max_replay_size,
                                 args.num_envs, state_dim, action_dim)
    manager = ManagerReplay.create(args.manager_replay_size if args.manager_enabled else 1,
                                   args.num_envs, observation_dim=observation_dim, action_dim=config.manager_action_dim)
    env_steps, iteration = 0, 0
    if payload is not None:
        agent = restore_agent(payload, agent)
        key = jnp.asarray(payload["key"])
        env_steps, iteration = payload["env_steps"], payload["iteration"]
        rollout = flax.serialization.from_state_dict(rollout, payload["rollout"])
        if "replay" in payload:
            worker = flax.serialization.from_state_dict(worker, payload["replay"]["worker"])
            manager = flax.serialization.from_state_dict(manager, payload["replay"]["manager"])

    collect = make_collector(env, args.unroll_length)
    learn = make_learner(args)
    eval_env = (wrap_env(args, evaluation=True)
                if args.goal_start_probability > 0 or (args.eval_env_id and args.eval_env_id != args.env_id) else env)
    evaluate = make_evaluator(eval_env, args.num_eval_envs, args.episode_length)
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
            agent, key, learning_metrics = learn(agent, worker, manager, key)
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
                metrics.update(evaluate(agent, jax.random.fold_in(key, 2)))
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
