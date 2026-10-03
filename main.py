"""Hydra entry point for single runs, local sweeps, and Submitit Slurm arrays."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def training_args(cfg: DictConfig, output_dir: str):
    """Translate Hydra run controls into the vectorized trainer's units."""
    from train_hierarchical import Args, resolve_puzzle_goal_config

    values = OmegaConf.to_container(cfg, resolve=True)
    agent = values.pop("agent")
    wandb = values.pop("wandb")
    values.pop("save_dir")
    values.pop("discount")  # Resolved through agent.manager_discount.
    task_id = values.pop("task_id", None)
    if values["env_id"].replace("_", "-") == "puzzle-3x3":
        if type(task_id) is not int or not 1 <= task_id <= 5:
            raise ValueError("Puzzle task_id must be an integer from 1 to 5.")
        values["env_id"] = f"puzzle-3x3-singletask-task{task_id}-v0"
    num_envs = values["num_envs"]

    def per_stream(total):
        return (total + num_envs - 1) // num_envs

    values["total_env_steps"] = values.pop("online_steps")
    values["num_eval_envs"] = values.pop("eval_episodes")
    values["max_replay_size"] = per_stream(values.pop("buffer_size"))
    values["manager_replay_size"] = per_stream(values.pop("manager_buffer_size"))
    values["min_replay_size"] = max(1, per_stream(values.pop("start_training")))
    if values.get("target") is not None:
        values["target"] = tuple(values["target"])
    values["output_dir"] = str(Path(output_dir).resolve())
    for name in ("resume", "worker_checkpoint"):
        values[name] = to_absolute_path(values[name]) if values[name] else ""
    for name in ("goal_low", "goal_high", "policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
        if agent[name] is not None:
            agent[name] = tuple(agent[name])
    return resolve_puzzle_goal_config(Args(**values, **agent, track=wandb["enabled"],
                wandb_project=wandb["project"], wandb_entity=wandb["entity"],
                wandb_mode=wandb["mode"]))


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    # Keep JAX/Brax imports in the job, so --cfg and Slurm submission are light.
    from train_hierarchical import main as train

    output_dir = HydraConfig.get().runtime.output_dir
    args = training_args(cfg, output_dir)
    cfg.agent.goal_low, cfg.agent.goal_high = list(args.goal_low), list(args.goal_high)
    OmegaConf.save(cfg, Path(output_dir) / "resolved_config.yaml", resolve=True)
    train(args, tracking_config=OmegaConf.to_container(cfg, resolve=True))
    # Submitit serializes the return value; do not return the agent/optimizer trees.


if __name__ == "__main__":
    main()
