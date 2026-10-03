"""Portable puzzle trajectories and offline rendering (no physics replay needed)."""

import json
from pathlib import Path

import numpy as np


class PuzzleRolloutRecorder:
    """Record small batched state arrays, keeping terminal physics before reset.

    States have shape [T+1, N, ...]; transition arrays have shape [T, N, ...].
    Use an episode wrapper without autoreset so final button presses are saved.
    """

    def __init__(self, state):
        self.states = [self._state(state)]
        self.transitions = []

    @staticmethod
    def _state(state):
        import jax
        return jax.device_get(dict(
            observations=state.obs, qpos=state.data.qpos, qvel=state.data.qvel,
            ctrl=state.data.ctrl, button_states=state.info["button_states"],
            target_button_states=state.info["target_button_states"],
        ))

    def append(self, state, actions, **extra):
        import jax
        self.states.append(self._state(state))
        self.transitions.append(jax.device_get(dict(
            actions=actions, rewards=state.reward, dones=state.done,
            truncations=state.info.get("truncation", np.zeros_like(state.done)),
            successes=state.metrics["success"], valid=state.metrics["valid"],
            ik_no_solution=state.metrics["ik_no_solution"], **extra,
        )))

    def save(self, path, model, metadata):
        import mujoco
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.transitions:
            raise ValueError("A rollout must contain at least one transition.")
        arrays = {key: np.stack([s[key] for s in self.states]) for key in self.states[0]}
        arrays.update({key: np.stack([t[key] for t in self.transitions]) for key in self.transitions[0]})
        model_path = path.with_suffix(".mjb")
        mujoco.mj_saveModel(model, str(model_path))
        metadata = dict(metadata, format_version=1, model_file=model_path.name,
                        mujoco_version=mujoco.__version__)
        np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
        summary = dict(metadata,
                       steps=len(self.transitions), num_envs=arrays["qpos"].shape[1],
                       success_rate=float(np.any(arrays["successes"], axis=0).mean()),
                       first_success_steps=[int(np.flatnonzero(s)[0] + 1) if np.any(s) else None
                                            for s in arrays["successes"].T],
                       valid_fraction=float(arrays["valid"].mean()),
                       final_button_states=arrays["button_states"][-1].tolist(),
                       target_button_states=arrays["target_button_states"][-1].tolist())
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def render_rollout(path, output=None, env_index=0, width=640, height=480, stride=1, gif=False):
    """Render stored physics and logical button state, on a CPU or GPU host."""
    import mediapy
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    path = Path(path)
    output = Path(output) if output else path.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        qpos, qvel = archive["qpos"], archive["qvel"]
        buttons, targets = archive["button_states"], archive["target_button_states"]
        button_goals = metadata.get("goal_mode") == "button_xy_depression"
        commands = archive["manager_goals"] if button_goals else None
    if not 0 <= env_index < qpos.shape[1]:
        raise ValueError(f"env_index must be between 0 and {qpos.shape[1] - 1}.")
    if min(width, height, stride) < 1:
        raise ValueError("Render dimensions and stride must be positive.")
    model = mujoco.MjModel.from_binary_path(str(path.parent / metadata["model_file"]))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    button_geoms = [model.geom(f"btngeom_{i}").id for i in range(9)]
    fps = 1.0 / (metadata["ctrl_dt"] * stride)
    frame_indices = list(range(0, len(qpos), stride))
    if frame_indices[-1] != len(qpos) - 1:
        frame_indices.append(len(qpos) - 1)
    sheet_indices = set(np.linspace(0, len(frame_indices) - 1, 6, dtype=int).tolist())
    sheet, animation = [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with mediapy.VideoWriter(str(output), shape=(height, width), fps=fps) as writer:
            for frame_index, t in enumerate(frame_indices):
                data.qpos[:] = qpos[t, env_index]
                data.qvel[:] = qvel[t, env_index]
                for geom, bit in zip(button_geoms, buttons[t, env_index]):
                    model.geom_rgba[geom] = [0.2, 0.2, 0.8, 1.] if bit else [0.8, 0.2, 0.2, 1.]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="front")
                frame = renderer.render().copy()
                # Make the task and actual logical state visible in the video.
                labeled = Image.fromarray(frame)
                draw = ImageDraw.Draw(labeled)
                current = ''.join(str(int(b)) for b in buttons[t, env_index])
                target = ''.join(str(int(b)) for b in targets[t, env_index])
                draw.rectangle((0, 0, width, 52 if button_goals else 36), fill="black")
                draw.text((8, 3), f"{metadata.get('policy', 'policy')} | step {t} | red=0 blue=1", fill="white")
                if button_goals:
                    goal = commands[min(t, len(commands) - 1), env_index]
                    label = f"command XY=({goal[0]:.3f}, {goal[1]:.3f}) m  depression={goal[2]:.2f}"
                else:
                    label = f"buttons {current}   target {target}"
                draw.text((8, 19), label, fill="white")
                if button_goals:
                    draw.text((8, 35), f"buttons {current}   task target {target}", fill="white")
                writer.add_image(np.asarray(labeled))
                if frame_index in sheet_indices:
                    sheet.append(labeled.resize((320, 240)))
                if gif and (frame_index % max(1, round(fps / 10)) == 0 or t == frame_indices[-1]):
                    animation.append(labeled.resize((320, 240)))
    montage = Image.new("RGB", (320 * 3, 240 * 2))
    for i, frame in enumerate(sheet):
        montage.paste(frame, (320 * (i % 3), 240 * (i // 3)))
    montage.save(output.with_suffix(".png"))
    if animation:
        animation[0].save(output.with_suffix(".gif"), save_all=True, append_images=animation[1:],
                          duration=1000 * max(1, round(fps / 10)) / fps, loop=0)
    return output
