# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities

<p align="center">
    <a href= "https://arxiv.org/abs/2503.14858">
        <img src="https://img.shields.io/badge/arXiv-2311.10090-b31b1b.svg" /></a>
    <a href= "https://github.com/wang-kevin3290/scaling-crl/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://wang-kevin3290.github.io/scaling-crl/">
        <img src="https://img.shields.io/badge/website-purple" /></a>
</p>

> [!IMPORTANT]  
> [Our work was selected for the Best Paper Award at NeurIPS 2025!](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/#:~:text=1000%20Layer%20Networks%20for%20Self%2DSupervised%20RL%3A%20Scaling%20Depth%20Can%20Enable%20New%20Goal%2DReaching%20Capabilities) 🥳

Email kw6487@princeton.edu with questions/comments/suggestions.

![Environments](assets/envs.gif)
Our work builds on top of [JAXGCRL](https://github.com/MichalBortkiewicz/JaxGCRL), feel free to check it out!!

# Installation

```sh
uv sync
```
Then just fix the two Brax issues described below, and you'll be all set.

On Linux, `uv sync` installs JAX 0.4.23's `cuda12_pip` extra, including the
CUDA runtime, compiler, and math libraries. cuDNN is constrained to version 8
to match this JAX wheel. Installing the CUDA-enabled `jaxlib` wheel alone
does not install these dependencies.


## Fixing two bugs in brax 0.10.1
1. There is a minor bug in brax's contact.py file. To fix it, first locate the brax contact.py file in your virtual environment: 
```
find .venv -name contact.py
```
Then open the file and replace it with the following code:
```python
from typing import Optional
from brax import math
from brax.base import Contact
from brax.base import System
from brax.base import Transform
import jax
from jax import numpy as jp
from mujoco import mjx

def get(sys: System, x: Transform) -> Optional[Contact]:
    """Calculates contacts.
    Args:
        sys: system defining the kinematic tree and other properties
        x: link transforms in world frame
    Returns:
        Contact pytree
    """
    #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
    # ncon = mjx.ncon(sys)
    # if not ncon:
    #   return None
    data = mjx.make_data(sys)
    if data.ncon == 0:
        return None
    @jax.vmap
    def local_to_global(pos1, quat1, pos2, quat2):
        pos = pos1 + math.rotate(pos2, quat1)
        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
        return pos, mat
    x = x.concatenate(Transform.zero((1,)))
    xpos = x.pos[sys.geom_bodyid - 1]
    xquat = x.rot[sys.geom_bodyid - 1]
    geom_xpos, geom_xmat = local_to_global(
        xpos, xquat, sys.geom_pos, sys.geom_quat
    )
    # pytype: disable=wrong-arg-types
    d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
    d = mjx.collision(sys, d)
    # pytype: enable=wrong-arg-types
    c = d.contact
    elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
    body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
    body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
    link_idx = (body1, body2)
    return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
```
2. There is also a minor bug in brax's json.py file. To fix it, first locate the brax json.py file in your virtual environment:
```
find .venv -name json.py | grep "/brax/io/json.py"
```
Then open the file and change the if statement in line 159 to:  
```python
if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
```


# Running experiments
Now, we are ready to run the train script. To run the code, you'll need a GPU. For Humanoid-based environments, it may require up to 80GB of GPU memory (for deep networks). Below is an example command to run the training script (an additional example can be found in the provided slurm script `job.slurm`): 

```sh
uv run train.py --env_id "humanoid" --eval_env_id "humanoid" --num_epochs 100 --total_env_steps 100000000 --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 --vis_length 1000 --save_buffer 0 
```


>[!NOTE]
>If you would like the experiments to be synced to wandb, you should go to `train.py` and replace the default values of `wandb_entity` and `wandb_project_name` (line 34-35 of the `train.py` file) with your particular wandb entity and wandb project name. Alternatively, these two can also be set as hyperparameter flags when running the train script.

# Hierarchical training on a fixed point-maze task

`train_hierarchical.py` trains a SAC manager and a Scaling-CRL worker jointly
from scratch in the local Brax point U-maze. Follow the installation and two
Brax fixes above first. The existing `train.py` entry point is unchanged in use.

The point starts at `(4, 4)` and the task target is fixed at `(12, 4)`. Reward
is 1 within radius 0.5 of that target and 0 elsewhere, with no success
termination. Episodes last 1,000 steps by default. The manager sees the full
observation (position, velocity, task target); the worker sees position,
velocity, and the manager's absolute XY command. Commands range over
`[2, 14]²` and are held for 25 steps, including across collection chunks.

The manager uses twin scalar BroNet critics (256 units, two residual blocks)
and their **mean** in both the TD target and actor loss. Its actor also learns
through the worker's action and entropy, with weight 1. Worker parameters stay
fixed during this manager gradient, and the score's goal-encoder input is
detached. The worker retains this repo's CRL networks and future-goal losses.
Both levels tune their own temperature; manager target entropy defaults to
`-0.5 * goal_dim = -1` in normalized manager-action coordinates.

The hierarchical entry point uses Hydra, with the same config-group and
Submitit launcher workflow as `dyna-mpo/main.py`. Run `uv sync` to install
`hydra-core` and `hydra-submitit-launcher`. `train_hierarchical.py` also accepts
the same Hydra arguments; the old `--flag value` CLI has been replaced by
`key=value` overrides. The original standalone `train.py` still uses Tyro.

A small smoke run (CPU is sufficient; first compilation takes time):

```sh
uv run python main.py +experiment=smoke save_dir=/tmp/scaling-crl-smoke
```

A full training run (GPU recommended):

```sh
uv run python main.py +experiment=pointmaze_hierarchical \
  seed=0 online_steps=100000000 save_dir=runs
```

A one-hour Euler learning check on the fixed-target U-maze (one seed):

```sh
source setup.bash
uv run python main.py -m +experiment=pointmaze_one_hour hydra/launcher=slurm \
  'hydra.launcher.setup=["source /cluster/home/yardas/scaling-crl/setup.bash", "export JAX_PLATFORMS=cuda", "export MUJOCO_GL=disable"]' \
  save_dir=/cluster/scratch/$USER/scaling-crl
```

This preset stops training after 55 minutes and performs final evaluation and
checkpointing, including replay, before the launcher's 60-minute limit.
`max_runtime_seconds` is checked at collection/update boundaries, so allow
headroom for one iteration and final evaluation. The hierarchical fixed-task
experiment checks learning progress; it does not reproduce the original
goal-conditioned CRL benchmark scores.

Set `agent.manager_enabled=false` for a worker-only ablation, for example add
`agent.manager_enabled=false run_group=pointmaze-worker-only` to the one-hour
command. Collection, evaluation, and rendering send the fixed environment task
goal directly to the worker. Manager inference, replay insertion/sampling,
updates, and gradient diagnostics are skipped. Worker future-goal relabeling
and entropy tuning are unchanged. The flag is saved in checkpoints and restored
on resume; start fresh to compare the two modes. Unused manager parameters stay
in the checkpoint but are never updated in worker-only mode.

### Ant Big Maze with our worker

`+experiment=ant_big_maze_worker` uses the same `main.py` worker-only path with
the local Brax `ant_big_maze` training layout and `ant_big_maze_eval` evaluation
layout. State/action dimensions are inferred from the environment (29 and 8).
The environment supplies each episode's goal directly to the worker; the
point-maze `target` setting does not apply. In worker-only mode, its dense
environment reward is logged but is not used in the CRL updates.

The preset follows the repository's `job.slurm`: 100M steps, depth 8, width 256,
batch size 512, 512 parallel environments, 1,000-step episodes, and 800 updates
per 62-step collection (approximately one update per 40 environment steps).
Replay holds 10,000 steps per environment, with 1,000-step warmup. It uses our
episode-filtered future-goal sampler and update loop, so this is a comparison
of our implementation at the reference settings, not an exact rerun of `train.py`.

```sh
source setup.bash
uv run python main.py -m +experiment=ant_big_maze_worker hydra/launcher=slurm \
  hydra.launcher.timeout_min=1440 hydra.launcher.signal_delay_s=30 \
  'hydra.launcher.setup=["source /cluster/home/yardas/scaling-crl/setup.bash", "export JAX_PLATFORMS=cuda", "export MUJOCO_GL=disable"]' \
  seed=0 save_dir=/cluster/scratch/$USER/scaling-crl
```

Evaluation uses 128 deterministic episodes. `eval/success_steps` is the mean
number of steps within the success radius, corresponding to the reference
trainer's `eval/episode_success`; `eval/success_rate` instead counts episodes
that ever succeed. `eval/return` is the environment reward and is a different
metric. Checkpoints include replay when `save_replay=true`.
The [paper](https://arxiv.org/html/2503.14858v4) uses five seeds for its main
depth-scaling curves. This preset sweeps seeds 0–4 in multirun mode; pass
`seed=0` for a single run or `seed=1,2,3,4` to add the remaining seeds.

For algorithm comparisons, use `eval/success_steps` (time at goal), not dense
`eval/return` or the fraction `eval/success_rate`. Following paper Section 4.1,
average the last five evaluations for each completed training seed, then report
the mean and standard error across seeds. Keep environment, network depth, and
training budget matched; the paper's 441 ± 25 Ant Big Maze table entry is for
depth 64, whereas this preset uses depth 8.

After downloading one metrics file per seed, generate the comparison with:

```sh
MPLCONFIGDIR=/tmp/scaling-crl-matplotlib uv run python scripts/compare_antmaze.py \
  runs/antmaze-paper-comparison/seed*.jsonl \
  --output runs/antmaze-paper-comparison/report
```

An optional `--reference path/to/curve.csv` overlays `env_steps,success_steps`
reference data. Figure-extracted data must be identified as approximate.
The report withholds the final five-seed score until all five supplied runs
reach 100M steps; partial curves remain available for monitoring.

The configs are organized as follows:

- `configs/main.yaml`: run budget, environment, replay, logging, W&B, and output paths.
- `configs/agent/hierarchical_sac.yaml`: worker and manager learning parameters.
- `configs/experiment/`: reusable experiment presets; `pointmaze_hierarchical`
  uses worker depth 16 and five seeds in multirun mode.
- `configs/hydra/launcher/slurm.yaml` and `configs/hardware/`: Submitit Slurm
  resources and the reference repo's RTX 4090/3090 hardware profiles.

Override learner settings with `agent.*`, for example
`agent.manager_worker_weight=0`, `agent.actor_depth=16`, or
`agent.manager_entropy_coefficient=0.5`. `discount` sets the manager's discount;
`agent.worker_discount` controls future-goal sampling independently. Network
depth retains the original CRL convention of four layers per residual block.
Use `--cfg job --resolve` to inspect the composed experiment without training.

### Sweeps and Slurm

On Euler, source your local `setup.bash` before launching. To also source it
inside each submitted job, pass
`hydra.launcher.setup=['source /cluster/home/yardas/scaling-crl/setup.bash']`
(quote the entire override in the shell). To require GPU execution, add
`export JAX_PLATFORMS=cuda` to that setup list; JAX will then fail explicitly
if CUDA is unavailable instead of falling back to CPU.

A local sweep runs the Cartesian product of the specified values:

```sh
uv run python main.py -m +experiment=smoke \
  seed=0,1 agent.manager_worker_weight=0,1
```

The worker-gradient ablation preset expands to 10 runs: five paired seeds at
weights 0 and 1. Submit it from the cluster with:

```sh
MUJOCO_GL=disable uv run python main.py -m \
  +experiment=pointmaze_worker_gradient \
  hydra/launcher=slurm +hardware=4090_rtx \
  hydra.launcher.timeout_min=240 hydra.launcher.array_parallelism=10 \
  save_dir=/cluster/scratch/$USER/scaling-crl \
  wandb.enabled=true wandb.mode=offline
```

`-m` is required to use the Submitit launcher. Without a launcher override,
Hydra runs the sweep locally. Use `hydra/launcher=submitit_local` to smoke-test
Submitit locally. Command-line sweep values override the preset's sweep values.

The Slurm profile matches the reference's Euler resources: one RTX 4090,
10 CPUs, 10 GiB per CPU, and account `ls_krausea`. Select `+hardware=3090_rtx`
or override `hydra.launcher.account`, `hydra.launcher.partition`,
`hydra.launcher.mem_per_cpu`, and `hydra.launcher.additional_parameters.gpus`
for another allocation. Timeouts default to 60 minutes; choose a suitable
limit for the experiment. Automatic timeout retries are disabled because the
launcher would restart the entry point; resume from a saved checkpoint instead.
The repository and installed virtual environment must be accessible to compute
nodes. Preview the launcher configuration without submitting:

```sh
uv run python main.py hydra/launcher=slurm +hardware=4090_rtx \
  --cfg hydra -p hydra.launcher
```

### Parameter units and saved runs

`online_steps`, `start_training`, `log_interval`, `eval_interval`, and
`save_interval` count **global primitive environment steps**. Collection is
batched, so budgets and interval events occur at the next collection boundary;
intervals need not divide the batch size. A nonpositive interval disables its
periodic event; final evaluation and checkpointing still run.

`buffer_size` and `manager_buffer_size` count total transitions across all
environments. They are rounded up to per-environment capacities internally.
`agent.batch_size` is the learner minibatch size. `eval_episodes` is the number
of parallel evaluation episodes. `updates_per_collect` counts joint learner
updates after each `num_envs * unroll_length` collection batch; it is deliberately
not called `utd_ratio`, since the reference's per-step update scheduling differs.
The default values retain this trainer's previous effective schedule.

Worker replay filters future goals by episode ID and uses the current state
when no future goal remains. Manager replay contains completed command intervals
and original manager actions. Autoreset timeouts mask the entire interval's
critic loss because its endpoint observation was replaced; true terminal
intervals retain their reward with zero bootstrap. Actor and temperature
updates can still use valid starting states from those intervals.

Hydra creates a separate directory for every job under `save_dir/hydra/` or
`save_dir/hydra/multirun/`, grouped by `run_group` and timestamp. Sweep subdirectories
include the job number and seed, so different settings with the same seed do
not overwrite each other. Each job writes `.hydra/` configs and overrides,
`resolved_config.yaml`, effective training `config.json`, `metrics.jsonl`, and
`checkpoint.pkl`. Submitit logs and job metadata live in the sweep's `.submitit/`
directory. Set `save_dir` to cluster scratch for experiments.

Metrics include return, success rate, final distance, losses, temperatures,
goal saturation, and manager gradient contributions. `wandb.enabled=true`
enables tracking with `wandb.project`, `wandb.entity`, and `wandb.mode`; runs
share the requested `run_group`. The composed Hydra configuration and effective
training settings are attached to each W&B run. Final visualization writes
`policy.html` and `manager_goals.npy`; disable it with `capture_vis=false`.

Checkpoints save both learners, optimizers, target critics, RNG, and counters.
With `save_replay=true`, they also save both buffers and the active rollout,
including unfinished command intervals. Resume into a fresh run directory:

```sh
uv run python main.py resume=/absolute/path/to/checkpoint.pkl \
  online_steps=200000000 save_dir=runs run_group=pointmaze-resumed
```

Relative checkpoint paths are resolved against the invocation directory even
when Hydra changes the working directory. Resume restores saved learning/environment
settings; run budget, logging/checkpoint intervals, output, replay export, and
visualization controls come from the new command. The effective restored
settings are recorded in `config.json`. Without saved replay, collection starts
fresh and warms up again while retaining the learner and counters.

### Train only the manager with a pretrained worker

Use `worker_checkpoint=/absolute/path/to/checkpoint.pkl` with
`agent.freeze_worker=true` to initialize a new manager run. This loads the worker
actor, both contrastive encoders, learned temperature, and their optimizer states.
Worker network widths, depths, and activation choice come from the checkpoint.
The manager, its target critics and optimizers, replay, RNG, and training counters
start fresh. A source worker-only checkpoint does not disable the new manager.

Freezing skips all worker optimizer/temperature updates and primitive replay
storage/sampling. The manager retains its auxiliary gradient through the worker's
actions and log probabilities; the goal-encoder input stays detached. Manager
replay warms up for `start_training` primitive steps before optimization.
`worker/frozen=1` is logged, together with the fixed worker temperature and manager
losses/gradient diagnostics. `agent.manager_worker_weight=0` disables the auxiliary
objective for a standard SAC manager ablation.

For Ant Big Maze, the manager preset uses the existing training/evaluation goal
distributions and 1,000-step episodes. The manager sees the full state and task
goal and outputs absolute XY commands in `[2, 26]²`, held for 25 steps. Its task
reward is 1 within distance 0.5 of the environment goal, 0 otherwise; reaching the
goal does not terminate the episode. Unhealthy-ant termination is unchanged.
Evaluation still reports `eval/success_steps` and `eval/success_rate`.

```sh
source setup.bash
uv run python main.py -m +experiment=ant_big_maze_manager hydra/launcher=slurm \
  worker_checkpoint=/absolute/path/to/worker/checkpoint.pkl \
  hydra.launcher.timeout_min=1440 hydra.launcher.signal_delay_s=30 \
  'hydra.launcher.setup=["source /cluster/home/yardas/scaling-crl/setup.bash", "export JAX_PLATFORMS=cuda", "export MUJOCO_GL=disable"]' \
  seed=0,1,2,3,4 save_dir=/cluster/scratch/$USER/scaling-crl
```

This trains five fresh managers against the same frozen worker. To study worker
seed variation, sweep `worker_checkpoint` as well. Each training run keeps a
rolling `checkpoint.pkl` (overwritten at save intervals and at completion);
copy the selected pretrained checkpoint to a stable path before launching a
manager sweep if worker training is still running.

Manager checkpoints include the frozen worker. Continue with `resume=...` alone;
the original worker checkpoint is no longer needed. `resume` and
`worker_checkpoint` are separate modes and cannot be passed together.

A one-off CPU smoke sequence (no maintained test suite):

```sh
JAX_PLATFORMS=cpu uv run python main.py +experiment=smoke \
  agent.manager_enabled=false hydra.run.dir=/tmp/crl-worker-smoke
JAX_PLATFORMS=cpu uv run python main.py +experiment=smoke \
  worker_checkpoint=/tmp/crl-worker-smoke/checkpoint.pkl agent.freeze_worker=true \
  hydra.run.dir=/tmp/crl-manager-smoke
JAX_PLATFORMS=cpu uv run python main.py +experiment=smoke \
  resume=/tmp/crl-manager-smoke/checkpoint.pkl online_steps=100 \
  hydra.run.dir=/tmp/crl-manager-resumed
```

Check worker parameter/optimizer equality across these checkpoints, changing
manager parameters, finite losses, nonzero worker-path manager gradients, empty
primitive replay, populated manager replay, and preserved freeze mode on resume.

# Citing Scaling CRL 📜
```bibtex
@inproceedings{wang2025,
  title     = {1000 Layer Networks for Self-Supervised {RL}: Scaling Depth Can Enable New Goal-Reaching Capabilities},
  author    = {Kevin Wang and Ishaan Javali and Micha{\l} Bortkiewicz and Tomasz Trzcinski and Benjamin Eysenbach},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=s0JVsx3bx1}
}
```



<!-- 
## Troubleshooting Potential Errors

**If you encounter the following error:**
```AttributeError: module 'mujoco.mjx' has no attribute 'ncon'```  

**Fix:**
1. Locate the brax contact.py file in your conda environment: 
   ```
   find ~/.conda/envs/scaling-crl -name contact.py
   ```
2. Open the file and replace it with the following code:

    ```python
    from typing import Optional
    from brax import math
    from brax.base import Contact
    from brax.base import System
    from brax.base import Transform
    import jax
    from jax import numpy as jp
    from mujoco import mjx

    def get(sys: System, x: Transform) -> Optional[Contact]:
        """Calculates contacts.
        Args:
            sys: system defining the kinematic tree and other properties
            x: link transforms in world frame
        Returns:
            Contact pytree
        """
        #NOTE: THIS WAS MODIFIED SINCE AFTER MUJOCO 3.1.5, mjx.ncon IS NOT AVAILABLE
        # ncon = mjx.ncon(sys)
        # if not ncon:
        #   return None
        data = mjx.make_data(sys)
        if data.ncon == 0:
            return None
        @jax.vmap
        def local_to_global(pos1, quat1, pos2, quat2):
            pos = pos1 + math.rotate(pos2, quat1)
            mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
            return pos, mat
        x = x.concatenate(Transform.zero((1,)))
        xpos = x.pos[sys.geom_bodyid - 1]
        xquat = x.rot[sys.geom_bodyid - 1]
        geom_xpos, geom_xmat = local_to_global(
            xpos, xquat, sys.geom_pos, sys.geom_quat
        )
        # pytype: disable=wrong-arg-types
        d = data.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
        d = mjx.collision(sys, d)
        # pytype: enable=wrong-arg-types
        c = d.contact
        elasticity = (sys.elasticity[c.geom1] + sys.elasticity[c.geom2]) * 0.5
        body1 = jp.array(sys.geom_bodyid)[c.geom1] - 1
        body2 = jp.array(sys.geom_bodyid)[c.geom2] - 1
        link_idx = (body1, body2)
        return Contact(elasticity=elasticity, link_idx=link_idx, **c.__dict__)
    ```
3. Save the file and rerun the training script.


**If you encounter the following error:** ```Error rendering final policy: unsupported operand type(s) for ==: 'ArrayImpl' and 'list'```  

**Fix:**
1. Locate the brax json.py file in your conda environment:
   ```
   find ~/.conda/envs/scaling-crl -name json.py | grep "/brax/io/json.py"
   ```
2. Open the file and change the if statement in line 159 to:
    ```python
    if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all():
    ```
3. Save the file and rerun the training script. -->
