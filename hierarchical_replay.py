"""JAX replay: primitive trajectories and completed manager intervals."""

from typing import Any

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class WorkerReplay:
    observations: Any
    actions: Any
    episode_ids: Any
    position: Any
    size: Any

    @classmethod
    def create(cls, capacity, num_envs, state_dim=4, action_dim=2):
        return cls(jnp.zeros((capacity, num_envs, state_dim)),
                   jnp.zeros((capacity, num_envs, action_dim)),
                   jnp.zeros((capacity, num_envs), jnp.int32),
                   jnp.int32(0), jnp.int32(0))

    def insert(self, observations, actions, episode_ids):
        capacity = self.observations.shape[0]
        return self.replace(
            observations=self.observations.at[self.position].set(observations),
            actions=self.actions.at[self.position].set(actions),
            episode_ids=self.episode_ids.at[self.position].set(episode_ids),
            position=(self.position + 1) % capacity,
            size=jnp.minimum(self.size + 1, capacity),
        )

    def sample(self, key, batch_size, discount, episode_length):
        """Same geometric future-goal distribution and self fallback as CRL.

        Logical indices preserve chronology when the storage ring wraps. Each
        environment has its own episode IDs; rows from other episodes, unwritten
        storage, and overwritten history cannot become positive goals.
        """
        time_key, env_key, goal_key = jax.random.split(key, 3)
        capacity, num_envs = self.episode_ids.shape
        times = jax.random.randint(time_key, (batch_size,), 0, self.size)
        envs = jax.random.randint(env_key, (batch_size,), 0, num_envs)
        oldest = self.position - self.size
        indices = (oldest + times) % capacity
        offsets = jnp.arange(min(episode_length, capacity))
        future_times = times[:, None] + offsets
        future_indices = (oldest + future_times) % capacity
        same_episode = self.episode_ids[future_indices, envs[:, None]] == self.episode_ids[indices, envs, None]
        valid = (future_times < self.size) & same_episode & (offsets > 0)
        log_weights = jnp.where(valid, offsets * jnp.log(discount), -jnp.inf)
        log_weights = log_weights.at[:, 0].set(jnp.log(1e-5))
        sampled_offsets = jax.random.categorical(goal_key, log_weights)
        goals = self.observations[(oldest + times + sampled_offsets) % capacity, envs, :2]
        return {"observations": self.observations[indices, envs],
                "actions": self.actions[indices, envs], "goals": goals}


@flax.struct.dataclass
class ManagerReplay:
    data: Any
    positions: Any
    sizes: Any

    @classmethod
    def create(cls, capacity, num_envs, observation_dim=6, goal_dim=2):
        shapes = {"observations": (observation_dim,), "actions": (goal_dim,),
                  "next_observations": (observation_dim,), "rewards": (),
                  "duration": (), "bootstrap": (), "valid": ()}
        data = {name: jnp.zeros((capacity, num_envs) + shape) for name, shape in shapes.items()}
        return cls(data, jnp.zeros(num_envs, jnp.int32), jnp.zeros(num_envs, jnp.int32))

    def insert(self, transitions, completed):
        """Append only completed intervals, independently for each environment."""
        capacity = self.data["rewards"].shape[0]
        envs = jnp.arange(self.sizes.shape[0])

        def insert_field(stored, new):
            mask = completed.reshape(completed.shape + (1,) * (new.ndim - 1))
            values = jnp.where(mask, new, stored[self.positions, envs])
            return stored.at[self.positions, envs].set(values)

        return self.replace(
            data=jax.tree_util.tree_map(insert_field, self.data, transitions),
            positions=(self.positions + completed.astype(jnp.int32)) % capacity,
            sizes=jnp.minimum(self.sizes + completed.astype(jnp.int32), capacity),
        )

    def sample(self, key, batch_size):
        env_key, time_key = jax.random.split(key)
        # Weight streams by their count so every stored interval is equiprobable.
        envs = jax.random.categorical(env_key, jnp.log(self.sizes.astype(jnp.float32)), shape=(batch_size,))
        indices = jax.random.randint(time_key, (batch_size,), 0, self.sizes[envs])
        return jax.tree_util.tree_map(lambda x: x[indices, envs], self.data)
