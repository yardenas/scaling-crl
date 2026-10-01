"""Endpoint capture and fresh episode resets for Brax training."""

from brax.envs.base import Wrapper
import jax
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


class ResamplingAutoResetWrapper(Wrapper):
    """Reset completed streams with fresh keys, preserving their terminal metrics.

    Use FinalObservationWrapper inside this wrapper to retain timeout endpoints.
    RNG and the new episode's start type live in checkpointed state.info.
    """

    def reset(self, rng):
        keys = jax.vmap(jax.random.split)(rng)
        state = self.env.reset(keys[:, 0])
        state.info["reset_rng"] = keys[:, 1]
        return state

    def step(self, state, action):
        state = state.replace(
            done=jnp.zeros_like(state.done),
            info={**state.info, "steps": jnp.where(state.done, 0, state.info["steps"])},
        )
        state = self.env.step(state, action)
        keys = jax.vmap(jax.random.split)(state.info["reset_rng"])

        def reset(_):
            fresh = self.env.reset(keys[:, 0])
            return fresh.pipeline_state, fresh.obs, fresh.info["goal_start"]

        # Avoid pipeline initialization when no stream ended this step.
        pipeline, obs, goal_start = jax.lax.cond(
            jnp.any(state.done), reset,
            lambda _: (state.pipeline_state, state.obs, state.info["goal_start"]),
            operand=None,
        )

        def where_done(fresh, current):
            done = state.done.reshape(state.done.shape + (1,) * (current.ndim - state.done.ndim))
            return jnp.where(done, fresh, current)

        state.info["goal_start"] = where_done(goal_start, state.info["goal_start"])
        state.info["reset_rng"] = keys[:, 1]
        return state.replace(
            pipeline_state=jax.tree_util.tree_map(where_done, pipeline, state.pipeline_state),
            obs=where_done(obs, state.obs),
        )
