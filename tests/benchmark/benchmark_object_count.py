"""
Script to benchmark speed vs. no. of objects in the scene.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from omnigibson import app, Simulator
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.scenes.scene_base import Scene
from omnigibson.utils.asset_utils import get_og_assets_version


# Params to be set as needed.
MAX_NUM_OBJS = 400      # Maximum no. of objects to add.
NUM_OBJS_PER_ITER = 20   # No. of objects to add per iteration.
NUM_STEPS_PER_ITER = 30  # No. of steps to take for each n of objects.
OBJ_SCALE = 0.05         # Object scale to be set appropriately to sim collisions.
RAND_POSITION = True    # True to randomize positions.
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

# Internal constants.
_N_PER_ROW = int(np.sqrt(MAX_NUM_OBJS))
_MIN_VAL = -2.0
_MAX_VAL = 2.0
_STEP_SIZE = (_MAX_VAL - _MIN_VAL) / _N_PER_ROW


def _get_position(obj_idx, is_random=False):
    if is_random:
        pos_arange = np.arange(_MIN_VAL, _MAX_VAL, step=0.1, dtype=np.float32)
        x, y, z = np.random.choice(pos_arange, size=3)
        return x, y, z
    x = _MIN_VAL + _STEP_SIZE * (obj_idx % _N_PER_ROW)
    y = _MIN_VAL + _STEP_SIZE * (obj_idx // _N_PER_ROW)
    return x, y, 0.1


def benchmark_scene(sim):
    assets_version = get_og_assets_version()
    print("assets_version", assets_version)

    scene = Scene(floor_plane_visible=True)
    sim.import_scene(scene)
    sim.play()

    xs = []
    ys = []
    yerrs = []
    for i in range(MAX_NUM_OBJS // NUM_OBJS_PER_ITER):
        new_objs = []
        for j in range(NUM_OBJS_PER_ITER):
            obj_idx = i * NUM_OBJS_PER_ITER + j
            obj = PrimitiveObject(
                prim_path=f"/World/obj{obj_idx}",
                primitive_type="Sphere",
                name=f"obj{obj_idx}",
                scale=OBJ_SCALE,
                visual_only=False,
            )
            sim.import_object(obj=obj, auto_initialize=False)
            # x, y, z = _get_position(obj_idx, RAND_POSITION)
            x, y = 0, 0
            z = 0.5 + j * OBJ_SCALE * 2.25
            obj.set_position(position=np.array([x, y, z]))
            new_objs.append(obj)

        # Take a step to initialize the new objects (done in _non_physics_step()).
        sim.step()
        step_freqs = []
        for _ in range(NUM_STEPS_PER_ITER):
            start = time.time()
            sim.step()
            end = time.time()
            step_freqs.append(1 / (end - start))

        xs.append(i * NUM_OBJS_PER_ITER)
        max_freq, min_freq = np.max(step_freqs), np.min(step_freqs)
        ys.append(np.mean((max_freq, min_freq)))
        yerrs.append(max_freq - ys[-1])

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    plt.errorbar(xs, ys, yerr=yerrs, elinewidth=0.75)
    ax.set_xlabel("No. of objects")
    ax.set_ylabel("Step fps")
    ax.set_title(f"Version {assets_version}")
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, f"scene_objs_benchmark_{MAX_NUM_OBJS}_{OBJ_SCALE}.png"))


def main():
    assert MAX_NUM_OBJS <= 1000

    sim = Simulator()
    benchmark_scene(sim)
    app.close()


if __name__ == "__main__":
    main()
