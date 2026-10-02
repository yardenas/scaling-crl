"""OGBench 3x3 Lights Out with instantaneous, perfect button presses."""
import numpy as np
import jax.numpy as jnp
from brax.envs.base import Env, State

# Row-major boards from OGBench PuzzleEnv.set_tasks (3x3).
TASKS = (
    ('000000000', '110101011'),
    ('111111111', '011111111'),
    ('010111010', '101010101'),
    ('010101010', '111111111'),
    ('111111111', '101101101'),
)


def toggle_matrix_3x3():
    rows = np.arange(9) // 3
    cols = np.arange(9) % 3
    return ((abs(rows[:, None] - rows) + abs(cols[:, None] - cols)) <= 1).astype(np.int32)


class PuzzleLogic(Env):
    """Continuous SAC scores choose argmax button; state is [board, fixed goal]."""
    action_size = 9
    observation_size = 18
    goal_size = 9
    backend = 'logic'

    def __init__(self, task_id=1, target=None, terminate_on_success=True):
        if type(task_id) is not int or not 1 <= task_id <= len(TASKS):
            raise ValueError('Puzzle task_id must be an integer from 1 to 5.')
        initial, goal = TASKS[task_id - 1]
        self.initial = jnp.array([int(b) for b in initial], dtype=jnp.float32)
        self.target = jnp.array([int(b) for b in goal] if target is None else target, dtype=jnp.float32)
        if self.target.shape != (9,) or not bool(jnp.all((self.target == 0) | (self.target == 1))):
            raise ValueError('Puzzle target must contain nine binary light states.')
        self.toggles = jnp.asarray(toggle_matrix_3x3())
        self.terminate_on_success = terminate_on_success

    def reset(self, rng):
        del rng
        obs = jnp.concatenate([self.initial, self.target])
        return State(pipeline_state=None, obs=obs, reward=jnp.float32(0), done=jnp.float32(0),
                     metrics={'success':jnp.float32(0), 'dist':jnp.sum(self.initial != self.target).astype(jnp.float32)})

    def step(self, state, action):
        button = jnp.argmax(action)
        board = jnp.bitwise_xor(state.obs[:9].astype(jnp.int32), self.toggles[button]).astype(jnp.float32)
        success = jnp.all(board == self.target).astype(jnp.float32)
        return state.replace(obs=jnp.concatenate([board,self.target]), reward=success - 1.,
                             done=success if self.terminate_on_success else jnp.float32(0),
                             metrics={'success':success,'dist':jnp.sum(board != self.target).astype(jnp.float32)})
