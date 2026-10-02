"""SAC manager with independent Scaling-CRL worker parameters and optimizers."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from crl_networks import Actor, G_encoder, SA_encoder

MANAGER_ACTIVATIONS = {"relu": nn.relu, "swish": nn.swish, "elu": nn.elu}


@dataclass(frozen=True)
class LearnerConfig:
    manager_enabled: bool = True
    freeze_worker: bool = False
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    actor_network_width: int = 256
    critic_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    use_relu: int = 0
    logsumexp_penalty_coeff: float = 0.1
    worker_discount: float = 0.99
    worker_entropy_coefficient: float = 0.5
    manager_width: int = 256
    manager_num_blocks: int = 2
    policy_hidden_layer_sizes: tuple[int, ...] | None = None
    value_hidden_layer_sizes: tuple[int, ...] | None = None
    manager_activation: str = "relu"
    manager_lr: float = 3e-4
    manager_discount: float = 0.99
    manager_discount_per_step: bool = True
    manager_tau: float = 0.005
    manager_init_temperature: float = 1.0
    manager_entropy_coefficient: float = 0.5
    manager_worker_weight: float = 1.0
    subgoal_steps: int = 25
    manager_learn_duration: bool = False
    max_subgoal_steps: int = 1000
    manager_count_bins: int = 16
    manager_count_bonus_scale: float = 0.0
    manager_action_candidates: int = 1
    goal_low: tuple[float, ...] = (2.0, 2.0)
    goal_high: tuple[float, ...] = (14.0, 14.0)

    @property
    def goal_dim(self):
        return len(self.goal_low)

    @property
    def manager_action_dim(self):
        return self.goal_dim + int(self.manager_learn_duration)


class ManagerActor(nn.Module):
    width: int = 256
    action_dim: int = 2
    activation: str = "relu"
    hidden_layer_sizes: tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, observations):
        activate = MANAGER_ACTIVATIONS[self.activation]
        x = observations
        for width in self.hidden_layer_sizes or (self.width, self.width):
            x = activate(nn.Dense(width)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = -5.0 + 3.5 * (nn.tanh(nn.Dense(self.action_dim)(x)) + 1.0)
        return mean, log_std


class BroCritic(nn.Module):
    """BroNet: Dense/LN/activation stem, residual blocks, scalar head.

    Architecture: github.com/naumix/BiggerRegularizedOptimistic
    (jaxrl/networks/common.py). No distributional or optimistic extensions.
    """

    width: int = 256
    num_blocks: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, observations, actions):
        activate = MANAGER_ACTIVATIONS[self.activation]
        dense = partial(nn.Dense, self.width,
                        kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))
        x = activate(nn.LayerNorm()(dense()(jnp.concatenate((observations, actions), -1))))
        for _ in range(self.num_blocks):
            residual = activate(nn.LayerNorm()(dense()(x)))
            x = x + nn.LayerNorm()(dense()(residual))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)[..., 0]


class ManagerCritics(nn.Module):
    width: int
    num_blocks: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, observations, actions):
        return jnp.stack([
            BroCritic(self.width, self.num_blocks, activation=self.activation, name=f"q{i}")(observations, actions)
            for i in range(2)
        ])


def sample_policy(apply_fn, params, observations, key, deterministic=False):
    mean, log_std = apply_fn(params, observations)
    noise = jax.random.normal(key, mean.shape)
    pre_tanh = mean if deterministic else mean + jnp.exp(log_std) * noise
    actions = jnp.tanh(pre_tanh)
    # Match the existing Scaling-CRL tanh Gaussian log probability.
    log_prob = jax.scipy.stats.norm.logpdf(pre_tanh, loc=mean, scale=jnp.exp(log_std))
    log_prob -= jnp.log(1.0 - actions ** 2 + 1e-6)
    return actions, log_prob.sum(-1)


def masked_mean(values, mask):
    return jnp.sum(jnp.where(mask > 0, values, 0.0) * mask) / jnp.maximum(mask.sum(), 1.0)


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(tree)))


@flax.struct.dataclass
class HierarchicalAgent:
    worker_actor: TrainState
    worker_critic: TrainState
    worker_alpha: TrainState
    manager_actor: TrainState
    manager_critic: TrainState
    manager_alpha: TrainState
    target_manager_critic: Any
    gradient_steps: Any
    config: LearnerConfig = flax.struct.field(pytree_node=False)
    sa_encoder: Any = flax.struct.field(pytree_node=False)
    g_encoder: Any = flax.struct.field(pytree_node=False)
    state_dim: int = flax.struct.field(pytree_node=False, default=4)
    action_dim: int = flax.struct.field(pytree_node=False, default=2)

    @classmethod
    def create(cls, key, config, observation_dim=6, state_dim=4, action_dim=2):
        keys = jax.random.split(key, 5)
        c = config
        actor = Actor(action_dim, network_width=c.actor_network_width,
                      network_depth=c.actor_depth, use_relu=c.use_relu)
        sa = SA_encoder(network_width=c.critic_network_width,
                        network_depth=c.critic_depth, use_relu=c.use_relu)
        goal = G_encoder(network_width=c.critic_network_width,
                         network_depth=c.critic_depth, use_relu=c.use_relu)
        manager_actor = ManagerActor(width=c.manager_width, action_dim=c.manager_action_dim,
                                     activation=c.manager_activation, hidden_layer_sizes=c.policy_hidden_layer_sizes)
        value_sizes = c.value_hidden_layer_sizes or (c.manager_width,) * c.manager_num_blocks
        if len(set(value_sizes)) != 1:
            raise ValueError("BroNet residual blocks must all have the same width.")
        manager_critic = ManagerCritics(value_sizes[0], len(value_sizes), activation=c.manager_activation)
        states, actions = jnp.ones((1, state_dim)), jnp.ones((1, action_dim))
        goals, observations = jnp.ones((1, c.goal_dim)), jnp.ones((1, observation_dim))

        def train_state(apply_fn, params, lr):
            return TrainState.create(apply_fn=apply_fn, params=params, tx=optax.adam(lr))

        critic_params = {
            "sa_encoder": sa.init(keys[1], states, actions),
            "g_encoder": goal.init(keys[2], goals),
        }
        manager_critic_params = manager_critic.init(keys[4], observations, jnp.ones((1, c.manager_action_dim)))
        return cls(
            worker_actor=train_state(actor.apply, actor.init(keys[0], jnp.concatenate((states, goals), -1)), c.actor_lr),
            worker_critic=train_state(None, critic_params, c.critic_lr),
            worker_alpha=train_state(None, {"log_alpha": jnp.float32(0)}, c.alpha_lr),
            manager_actor=train_state(manager_actor.apply, manager_actor.init(keys[3], observations), c.manager_lr),
            manager_critic=train_state(manager_critic.apply, manager_critic_params, c.manager_lr),
            manager_alpha=train_state(None, {"log_alpha": jnp.log(jnp.float32(c.manager_init_temperature))}, c.manager_lr),
            target_manager_critic=manager_critic_params,
            gradient_steps=jnp.int32(0), config=c, sa_encoder=sa, g_encoder=goal,
            state_dim=state_dim, action_dim=action_dim,
        )

    def goals(self, manager_actions):
        low, high = jnp.asarray(self.config.goal_low), jnp.asarray(self.config.goal_high)
        return low + (manager_actions[..., :self.config.goal_dim] + 1.0) * (high - low) / 2.0

    def manager_actions(self, observations, key, deterministic=False):
        return sample_policy(self.manager_actor.apply_fn, self.manager_actor.params,
                             observations, key, deterministic)[0]

    def count_indices(self, observations, actions):
        """Flatten bins of (position XY, commanded XY); duration is not counted.

        Position uses the same bounds as commands, clipping excursions to edge
        cells. Leading batch/candidate dimensions follow NumPy broadcasting.
        """
        bins = self.config.manager_count_bins
        low, high = jnp.asarray(self.config.goal_low), jnp.asarray(self.config.goal_high)
        position = (observations[..., :2] - low) / (high - low)
        command = (actions[..., :2] + 1.0) / 2.0
        position = jnp.clip(jnp.floor(position * bins), 0, bins - 1).astype(jnp.int32)
        command = jnp.clip(jnp.floor(command * bins), 0, bins - 1).astype(jnp.int32)
        return ((position[..., 0] * bins + position[..., 1]) * bins ** 2
                + command[..., 0] * bins + command[..., 1])

    def count_bonus(self, counts, observations, actions):
        if counts is None or self.config.manager_count_bonus_scale == 0:
            return jnp.zeros(actions.shape[:-1])
        visits = counts[self.count_indices(observations, actions)]
        return self.config.manager_count_bonus_scale * jax.lax.rsqrt(1.0 + visits)

    def exploratory_manager_actions(self, observations, key, counts):
        """Collection-only best-of-N; SAC still learns the Gaussian policy."""
        candidates = self.config.manager_action_candidates
        if candidates == 1:
            return self.manager_actions(observations, key)
        actions = jax.vmap(lambda k: self.manager_actions(observations, k))(
            jax.random.split(key, candidates))
        candidate_obs = jnp.broadcast_to(observations, (candidates,) + observations.shape)
        q = self.manager_critic.apply_fn(self.manager_critic.params, candidate_obs, actions).mean(0)
        scores = q + self.count_bonus(counts, observations, actions)
        return actions[jnp.argmax(scores, axis=0), jnp.arange(observations.shape[0])]

    def worker_actions(self, observations, goals, key, deterministic=False):
        inputs = jnp.concatenate((observations[..., :self.state_dim], goals), -1)
        return sample_policy(self.worker_actor.apply_fn, self.worker_actor.params,
                             inputs, key, deterministic)[0]

    def worker_score(self, observations, actions, goals):
        params = self.worker_critic.params
        sa = self.sa_encoder.apply(params["sa_encoder"], observations, actions)
        goal = self.g_encoder.apply(params["g_encoder"], goals)
        return -jnp.sqrt(jnp.sum((sa - goal) ** 2, axis=-1))

    def worker_actor_loss(self, params, batch, key):
        inputs = jnp.concatenate((batch["observations"], batch["goals"]), -1)
        actions, log_prob = sample_policy(self.worker_actor.apply_fn, params, inputs, key)
        score = self.worker_score(batch["observations"], actions, batch["goals"])
        loss = jnp.mean(jnp.exp(self.worker_alpha.params["log_alpha"]) * log_prob - score)
        return loss, (log_prob, score.mean())

    def worker_critic_loss(self, params, batch):
        sa = self.sa_encoder.apply(params["sa_encoder"], batch["observations"], batch["actions"])
        goals = self.g_encoder.apply(params["g_encoder"], batch["goals"])
        logits = -jnp.sqrt(jnp.sum((sa[:, None] - goals[None]) ** 2, axis=-1))
        loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
        regularizer = jnp.mean(jax.nn.logsumexp(logits + 1e-6, axis=1) ** 2)
        return loss + self.config.logsumexp_penalty_coeff * regularizer

    def manager_actor_terms(self, params, observations, key):
        high_key, low_key = jax.random.split(key)
        manager_actions, log_prob = sample_policy(self.manager_actor.apply_fn, params, observations, high_key)
        q = self.manager_critic.apply_fn(self.manager_critic.params, observations, manager_actions).mean(0)
        if self.config.manager_worker_weight == 0:
            return log_prob, -q, jnp.zeros_like(log_prob)
        goals = self.goals(manager_actions)
        states = observations[..., :self.state_dim]
        inputs = jnp.concatenate((states, goals), -1)
        actions, worker_log_prob = sample_policy(self.worker_actor.apply_fn, self.worker_actor.params, inputs, low_key)
        score = self.worker_score(states, actions, jax.lax.stop_gradient(goals))
        worker_loss = jnp.exp(self.worker_alpha.params["log_alpha"]) * worker_log_prob - score
        return log_prob, -q, self.config.manager_worker_weight * worker_loss

    def manager_actor_loss(self, params, batch, key):
        log_prob, q_loss, worker_loss = self.manager_actor_terms(params, batch["observations"], key)
        loss = jnp.mean(jnp.exp(self.manager_alpha.params["log_alpha"]) * log_prob + q_loss + worker_loss)
        return loss, (log_prob, q_loss.mean(), worker_loss.mean())

    def manager_critic_loss(self, params, batch, key):
        # Masked timeout endpoints are reset observations: don't feed them to the backup.
        bootstrap = batch["bootstrap"] * batch["valid"]
        next_observations = jnp.where(bootstrap[:, None] > 0, batch["next_observations"], 0.0)
        actions, log_prob = sample_policy(self.manager_actor.apply_fn, self.manager_actor.params, next_observations, key)
        q = self.manager_critic.apply_fn(self.target_manager_critic, next_observations, actions).mean(0)
        soft_q = q - jnp.exp(self.manager_alpha.params["log_alpha"]) * log_prob
        discount = (self.config.manager_discount ** batch["duration"]
                    if self.config.manager_discount_per_step else self.config.manager_discount)
        target = batch["rewards"] + discount * jnp.where(bootstrap > 0, soft_q, 0.0)
        qs = self.manager_critic.apply_fn(params, batch["observations"], batch["actions"])
        loss = masked_mean(jnp.mean((qs - jax.lax.stop_gradient(target)) ** 2, axis=0), batch["valid"])
        return loss, masked_mean(target, batch["valid"])

    @jax.jit
    def update(self, worker_batch, manager_batch, key):
        worker_key, actor_key, critic_key = jax.random.split(key, 3)
        agent = self.replace(gradient_steps=self.gradient_steps + 1)
        metrics = {
            "worker/frozen": jnp.float32(self.config.freeze_worker),
            "worker/temperature": jnp.exp(self.worker_alpha.params["log_alpha"]),
        }
        if not self.config.freeze_worker:
            (wa_loss, (worker_log_prob, score)), wa_grad = jax.value_and_grad(self.worker_actor_loss, has_aux=True)(self.worker_actor.params, worker_batch, worker_key)
            wc_loss, wc_grad = jax.value_and_grad(self.worker_critic_loss)(self.worker_critic.params, worker_batch)

            def worker_alpha_loss(params):
                target_entropy = -self.config.worker_entropy_coefficient * self.action_dim
                return jnp.exp(params["log_alpha"]) * jnp.mean(jax.lax.stop_gradient(-worker_log_prob - target_entropy))

            wal, wag = jax.value_and_grad(worker_alpha_loss)(self.worker_alpha.params)
            agent = agent.replace(
                worker_actor=self.worker_actor.apply_gradients(grads=wa_grad),
                worker_critic=self.worker_critic.apply_gradients(grads=wc_grad),
                worker_alpha=self.worker_alpha.apply_gradients(grads=wag),
            )
            metrics.update({
                "worker/actor_loss": wa_loss, "worker/critic_loss": wc_loss,
                "worker/alpha_loss": wal, "worker/score": score,
                "worker/temperature": jnp.exp(agent.worker_alpha.params["log_alpha"]),
                "worker/entropy": -worker_log_prob.mean(),
            })
        if not self.config.manager_enabled:
            return agent, metrics

        (ma_loss, (manager_log_prob, q_loss, worker_loss)), ma_grad = jax.value_and_grad(self.manager_actor_loss, has_aux=True)(self.manager_actor.params, manager_batch, actor_key)
        (mc_loss, target), mc_grad = jax.value_and_grad(self.manager_critic_loss, has_aux=True)(self.manager_critic.params, manager_batch, critic_key)

        def manager_alpha_loss(params):
            target_entropy = -self.config.manager_entropy_coefficient * self.config.manager_action_dim
            return -params["log_alpha"] * jnp.mean(jax.lax.stop_gradient(manager_log_prob + target_entropy))

        mal, mag = jax.value_and_grad(manager_alpha_loss)(self.manager_alpha.params)
        # No valid TD rows means no optimizer step, including Adam momentum.
        manager_critic = jax.lax.cond(
            jnp.any(manager_batch["valid"] > 0),
            lambda _: self.manager_critic.apply_gradients(grads=mc_grad),
            lambda _: self.manager_critic, None,
        )
        agent = agent.replace(
            manager_actor=self.manager_actor.apply_gradients(grads=ma_grad),
            manager_critic=manager_critic,
            manager_alpha=self.manager_alpha.apply_gradients(grads=mag),
            target_manager_critic=optax.incremental_update(manager_critic.params, self.target_manager_critic, self.config.manager_tau),
        )
        return agent, {**metrics,
            "manager/actor_loss": ma_loss, "manager/critic_loss": mc_loss,
            "manager/alpha_loss": mal, "manager/target": target,
            "manager/q_loss": q_loss, "manager/worker_loss": worker_loss,
            "manager/temperature": jnp.exp(agent.manager_alpha.params["log_alpha"]),
            "manager/entropy": -manager_log_prob.mean(),
            "manager/critic_valid_fraction": manager_batch["valid"].mean(),
        }

    @jax.jit
    def gradient_diagnostics(self, observations, key):
        def paths(params):
            _, q_loss, worker_loss = self.manager_actor_terms(params, observations, key)
            return jnp.stack((q_loss.mean(), worker_loss.mean()))
        gradients = jax.jacrev(paths)(self.manager_actor.params)
        q_grads = jax.tree_util.tree_map(lambda x: x[0], gradients)
        worker_grads = jax.tree_util.tree_map(lambda x: x[1], gradients)
        q_norm, worker_norm = tree_norm(q_grads), tree_norm(worker_grads)
        dot = sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(q_grads), jax.tree_util.tree_leaves(worker_grads)))
        return {"manager/q_path_grad_norm": q_norm,
                "manager/worker_path_grad_norm": worker_norm,
                "manager/path_grad_cosine": dot / jnp.maximum(q_norm * worker_norm, 1e-12)}
