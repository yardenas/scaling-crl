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

A small smoke run (CPU is sufficient; first compilation takes time):

```sh
uv run train_hierarchical.py \
  --num-envs 2 --num-eval-envs 2 --episode-length 16 \
  --subgoal-steps 3 --unroll-length 5 --min-replay-size 5 \
  --max-replay-size 24 --manager-replay-size 8 --batch-size 8 \
  --updates-per-collect 1 --total-env-steps 80 \
  --actor-network-width 16 --critic-network-width 16 \
  --manager-width 16 --manager-num-blocks 1 \
  --log-every 2 --eval-every 4 --save-replay --no-capture-vis \
  --output-dir runs/hierarchical_smoke
```

A full training run (GPU recommended):

```sh
uv run train_hierarchical.py \
  --total-env-steps 100000000 --num-envs 128 \
  --actor-depth 16 --critic-depth 16 \
  --output-dir runs/hierarchical_pointmaze
```

Use `--help` for all settings. `--manager-worker-weight 0` disables the extra
worker-gradient term, `--manager-entropy-coefficient` controls the manager
entropy target, and `--worker-discount` controls future-goal sampling separately
from `--manager-discount`. Network depth retains the original CRL convention
of four layers per residual block. Replay capacities and warmup are per
environment; `--updates-per-collect` is the number of joint learner updates
after each `num_envs * unroll_length` collection batch. The step budget rounds
up to a complete collection batch.

Worker replay filters future goals by episode ID and uses the current state
when no future goal remains. Manager replay contains only completed command
intervals and their original manager actions. Autoreset timeouts mask the
entire interval's critic loss because its endpoint observation was replaced;
true terminal intervals retain their reward with zero bootstrap. Actor and
temperature updates can still use the valid starting states of these intervals.

Each run writes `config.json`, `metrics.jsonl`, and `checkpoint.pkl`. Metrics
include return, success rate, final distance, losses, temperatures, goal
saturation, and the manager's Q and worker gradient contributions. Default
final visualization writes `policy.html` and `manager_goals.npy` using the same
goal commitment as training. Add `--track` to enable W&B (offline by default).

Checkpoints save both learners, optimizers, target critics, RNG, and counters.
With `--save-replay`, they also save both buffers and the active rollout,
including unfinished command intervals. Resume with:

```sh
uv run train_hierarchical.py \
  --resume runs/hierarchical_pointmaze/checkpoint.pkl \
  --total-env-steps 200000000 --output-dir runs/hierarchical_pointmaze_resumed
```

Resume restores the saved learning/environment configuration; run budget,
logging intervals, output, replay export, and visualization controls come from
the new command. Without saved replay, collection starts fresh and warms up
again while retaining learned parameters, optimizer states, and counters.

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
