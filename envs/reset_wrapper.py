"""Endpoint capture for Brax training."""

from brax.envs.base import Wrapper
import jax.numpy as jnp


class FinalObservationWrapper(Wrapper):
    """Capture endpoints inside autoreset, outside Vmap/EpisodeWrapper."""

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(info={**state.info,
                                  "final_observation": jnp.zeros_like(state.obs),
                                  "final_observation_valid": jnp.zeros_like(state.done, dtype=bool)})

    def step(self, state, action):
        state = self.env.step(state.replace(info=dict(state.info), metrics=dict(state.metrics)), action)
        return state.replace(info={**state.info, "final_observation": state.obs,
                                  "final_observation_valid": state.done.astype(bool)})
