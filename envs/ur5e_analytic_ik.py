"""Pure JAX analytic IK helpers for the OGBench UR5e attachment site."""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco.mjx._src import math as mjx_math

D1 = 0.163
D4 = 0.134
D5 = 0.1
D6 = 0.1
A2 = -0.425
A3 = -0.392
IK_EPS = 1e-9
IK_FK_TOL = 1e-3
WORLD_TO_DH_ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
DH_TO_WORLD_ROT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def wrap_to_pi(angle: jax.Array) -> jax.Array:
    return jnp.mod(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def _safe_sqrt(value: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(value, 0.0))


def _dh_matrix(
    theta: jax.Array,
    d: float | jax.Array,
    a: float | jax.Array,
    alpha: float | jax.Array,
) -> jax.Array:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    ca = jnp.cos(alpha)
    sa = jnp.sin(alpha)
    return jnp.array(
        [
            [c, -s * ca, s * sa, a * c],
            [s, c * ca, -c * sa, a * s],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )


def fk_dh(qpos: jax.Array) -> jax.Array:
    """Returns the UR5e DH-frame flange pose for six arm joints."""
    pose = jnp.eye(4, dtype=jnp.float32)
    pose = pose @ _dh_matrix(qpos[0], D1, 0.0, jnp.pi / 2.0)
    pose = pose @ _dh_matrix(qpos[1], 0.0, A2, 0.0)
    pose = pose @ _dh_matrix(qpos[2], 0.0, A3, 0.0)
    pose = pose @ _dh_matrix(qpos[3], D4, 0.0, jnp.pi / 2.0)
    pose = pose @ _dh_matrix(qpos[4], D5, 0.0, -jnp.pi / 2.0)
    return pose @ _dh_matrix(qpos[5], D6, 0.0, 0.0)


def fk_world_attach(qpos: jax.Array) -> jax.Array:
    """Returns the MuJoCo world-frame attachment-site pose for six arm joints."""
    dh_pose = fk_dh(qpos)
    dh_to_world = jnp.asarray(DH_TO_WORLD_ROT)
    world_pose = jnp.eye(4, dtype=jnp.float32)
    world_pose = world_pose.at[:3, :3].set(dh_to_world @ dh_pose[:3, :3])
    return world_pose.at[:3, 3].set(dh_to_world @ dh_pose[:3, 3])


def world_attach_pose_to_dh(target_pos: jax.Array, target_quat: jax.Array) -> jax.Array:
    world_to_dh = jnp.asarray(WORLD_TO_DH_ROT)
    world_rot = mjx_math.quat_to_mat(mjx_math.normalize(target_quat))
    dh_pose = jnp.eye(4, dtype=jnp.float32)
    dh_pose = dh_pose.at[:3, :3].set(world_to_dh @ world_rot)
    return dh_pose.at[:3, 3].set(world_to_dh @ target_pos)


def _calculate_theta6(
    sign5: jax.Array,
    c1: jax.Array,
    s1: jax.Array,
    r11: jax.Array,
    r12: jax.Array,
    r21: jax.Array,
    r22: jax.Array,
) -> jax.Array:
    h1 = c1 * r22 - s1 * r12
    h2 = s1 * r11 - c1 * r21
    h1 = jnp.where(jnp.abs(h1) < IK_EPS, 0.0, h1)
    h2 = jnp.where(jnp.abs(h2) < IK_EPS, 0.0, h2)
    return jnp.arctan2(sign5 * h1, sign5 * h2)


def _sign_or(value: jax.Array, fallback: float) -> jax.Array:
    sign = value / jnp.maximum(jnp.abs(value), IK_EPS)
    return jnp.where(jnp.abs(value) <= IK_EPS, fallback, sign)


def inverse_kinematics_dh(target_pose: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns 8 UR5e IK candidates plus a validity mask.

    Analytic equations adapted from:
    https://github.com/Victorlouisdg/ur-analytic-ik
    https://raw.githubusercontent.com/Victorlouisdg/ur-analytic-ik/main/src/ur_analytic_ik/inverse_kinematics.hh
    """

    r11, r12, r13 = target_pose[0, 0], target_pose[0, 1], target_pose[0, 2]
    r21, r22, r23 = target_pose[1, 0], target_pose[1, 1], target_pose[1, 2]
    r31 = target_pose[2, 0]
    px, py, pz = target_pose[0, 3], target_pose[1, 3], target_pose[2, 3]

    solutions = jnp.zeros((8, 6), dtype=jnp.float32)
    valid = jnp.ones((8,), dtype=jnp.bool_)

    a1 = px - D6 * r13
    b1 = D6 * r23 - py
    theta1_domain = a1 * a1 + b1 * b1 - D4 * D4
    theta1_valid = theta1_domain >= -IK_EPS
    theta1_delta = jnp.arctan2(_safe_sqrt(theta1_domain), D4)
    theta1_base = jnp.arctan2(a1, b1)
    theta1a = theta1_base + theta1_delta
    theta1b = theta1_base - theta1_delta
    solutions = solutions.at[:4, 0].set(theta1a)
    solutions = solutions.at[4:, 0].set(theta1b)
    valid = valid & theta1_valid

    for row in (0, 4):
        theta1 = solutions[row, 0]
        c1 = jnp.cos(theta1)
        s1 = jnp.sin(theta1)

        c5 = s1 * r13 - c1 * r23
        s5 = _safe_sqrt((s1 * r11 - c1 * r21) ** 2 + (s1 * r12 - c1 * r22) ** 2)
        theta5a = jnp.arctan2(s5, c5)
        theta5b = jnp.arctan2(-s5, c5)

        s5a = jnp.sin(theta5a)
        s5b = jnp.sin(theta5b)
        sign5a = _sign_or(s5a, 1.0)
        sign5b = jnp.where(jnp.abs(s5a) <= IK_EPS, -1.0, _sign_or(s5b, -1.0))

        theta6a = _calculate_theta6(sign5a, c1, s1, r11, r12, r21, r22)
        theta6b = _calculate_theta6(sign5b, c1, s1, r11, r12, r21, r22)

        solutions = solutions.at[row : row + 2, 4].set(theta5a)
        solutions = solutions.at[row : row + 2, 5].set(theta6a)
        solutions = solutions.at[row + 2 : row + 4, 4].set(theta5b)
        solutions = solutions.at[row + 2 : row + 4, 5].set(theta6b)

    for row in (0, 2, 4, 6):
        theta1 = solutions[row, 0]
        theta5 = solutions[row, 4]
        theta6 = solutions[row, 5]

        c1 = jnp.cos(theta1)
        s1 = jnp.sin(theta1)
        c5 = jnp.cos(theta5)
        s5 = jnp.sin(theta5)
        c6 = jnp.cos(theta6)
        s6 = jnp.sin(theta6)

        a234 = c1 * r11 + s1 * r21
        h1 = c5 * c6 * r31 - s6 * a234
        h2 = c5 * c6 * a234 + s6 * r31
        theta234 = jnp.arctan2(h1, h2)
        c234 = jnp.cos(theta234)
        s234 = jnp.sin(theta234)

        kc = c1 * px + s1 * py - s234 * D5 + c234 * s5 * D6
        ks = pz - D1 + c234 * D5 + s234 * s5 * D6
        c3 = (ks * ks + kc * kc - A2 * A2 - A3 * A3) / (2.0 * A2 * A3)
        theta3_domain = 1.0 - c3 * c3
        theta3_valid = theta3_domain >= -IK_EPS
        theta3a = jnp.arctan2(_safe_sqrt(theta3_domain), c3)
        theta3b = -theta3a

        c3a = jnp.cos(theta3a)
        s3a = jnp.sin(theta3a)
        theta2a = jnp.arctan2(ks, kc) - jnp.arctan2(A3 * s3a, A3 * c3a + A2)

        c3b = jnp.cos(theta3b)
        s3b = jnp.sin(theta3b)
        theta2b = jnp.arctan2(ks, kc) - jnp.arctan2(A3 * s3b, A3 * c3b + A2)

        solutions = solutions.at[row, 1:4].set(
            jnp.array([theta2a, theta3a, theta234 - theta2a - theta3a], dtype=jnp.float32)
        )
        solutions = solutions.at[row + 1, 1:4].set(
            jnp.array([theta2b, theta3b, theta234 - theta2b - theta3b], dtype=jnp.float32)
        )
        valid = valid.at[row : row + 2].set(valid[row : row + 2] & theta3_valid)

    solutions = wrap_to_pi(solutions)
    fk_poses = jax.vmap(fk_dh)(solutions)
    fk_error = jnp.max(jnp.abs(fk_poses - target_pose), axis=(1, 2))
    valid = valid & jnp.isfinite(solutions).all(axis=1)
    valid = valid & jnp.isfinite(fk_poses).all(axis=(1, 2))
    valid = valid & (fk_error <= IK_FK_TOL)
    return solutions, valid


def closest_ik_solution(
    solutions: jax.Array,
    valid: jax.Array,
    qpos0: jax.Array,
) -> jax.Array:
    qpos0_mapped = wrap_to_pi(qpos0)
    circular_diff = jnp.abs(solutions - qpos0_mapped)
    circular_diff = jnp.where(circular_diff > jnp.pi, 2.0 * jnp.pi - circular_diff, circular_diff)
    distances = jnp.sum(circular_diff, axis=1)
    distances = jnp.where(valid, distances, jnp.finfo(distances.dtype).max)
    closest = solutions[jnp.argmin(distances)]

    alternative = jnp.where(closest < 0.0, closest + 2.0 * jnp.pi, closest - 2.0 * jnp.pi)
    return jnp.where(jnp.abs(closest - qpos0) < jnp.abs(alternative - qpos0), closest, alternative)


def solve_ik_with_status(
    qpos0: jax.Array,
    target_pos: jax.Array,
    target_quat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    target_pose_dh = world_attach_pose_to_dh(target_pos, target_quat)
    solutions, valid = inverse_kinematics_dh(target_pose_dh)
    qpos_target = closest_ik_solution(solutions, valid, qpos0)
    no_soln = jnp.logical_not(jnp.any(valid))
    no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(target_pose_dh).all()))
    no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(qpos_target).all()))
    qpos_target = jnp.where(no_soln, qpos0, qpos_target)
    no_soln = jnp.logical_or(no_soln, jnp.logical_not(jnp.isfinite(qpos_target).all()))
    return qpos_target, no_soln


def solve_ik(qpos0: jax.Array, target_pos: jax.Array, target_quat: jax.Array) -> jax.Array:
    qpos_target, _ = solve_ik_with_status(qpos0, target_pos, target_quat)
    return qpos_target
