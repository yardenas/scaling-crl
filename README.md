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
