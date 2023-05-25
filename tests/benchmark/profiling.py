import os
import argparse
import json
import omnigibson as og
import numpy as np
import omnigibson.utils.transform_utils as T

from omnigibson.macros import gm
from omnigibson.systems import get_system
from omnigibson.object_states import Covered
from omnigibson.utils.profiling_utils import ProfilingEnv
from omnigibson.utils.constants import PrimType

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--robot", action='store_true')
parser.add_argument("-s", "--scene")
parser.add_argument("-f", "--flatcache", action='store_true')
parser.add_argument("-c", "--softbody", action='store_true')
parser.add_argument("-w", "--fluids", action='store_true')
parser.add_argument("-p", "--macro_particle_system", action='store_true')

PROFILING_FIELDS = ["Total frame time", "Physics step time", "Render step time", "Non-physics step time"]

def main():
    args = parser.parse_args()
    # Modify flatcache, pathtracing, GPU, and object state settings
    assert not (args.flatcache and args.softbody), "Flatcache cannot be used with softbody at the same time"
    gm.ENABLE_FLATCACHE = args.flatcache
    gm.ENABLE_HQ_RENDERING = args.fluids
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.USE_GPU_DYNAMICS = True

    cfg = {}
    if args.robot:
        cfg["robots"] =  [{
            "type": "Fetch",
            "obs_modalities": ["scan", "rgb", "depth"],
            "action_type": "continuous",
            "action_normalize": True,
            "position": [-1.3, 0, 0],
            "orientation": [0., 0., 0.7071, -0.7071]
        }]

    if args.scene:
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    else:
        cfg["scene"] = {
            "type": "Scene",
        }

    cfg["objects"] = [{
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "rjgmmy",
        "scale": 0.75,
        "position": [0, 0, 0.5],
    }]
    
    if args.softbody:
        cfg["objects"].append({
            "type": "DatasetObject",
            "name": "shirt",
            "category": "t_shirt",
            "model": "kvidcx",
            "prim_type": PrimType.CLOTH,
            "scale": 0.03,
            "position": [-0.4, -1, 1.5],
            "orientation": [0.7071, 0., 0.7071, 0.],
        })
    
    cfg["objects"].extend([{
        "type": "DatasetObject",
        "name": "apple",
        "category": "apple",
        "model": "agveuv",
        "scale": 1.5,
        "position": [0, 0, 0.8],
        "abilities": {"diceable": {}} if args.macro_particle_system else {}
    },
    {
        "type": "DatasetObject",
        "name": "knife",
        "category": "table_knife",
        "model": "lrdmpf",
        "scale": 2.5,
        "position": [0, 0, 1.2],
        "orientation": T.euler2quat([-np.pi / 2, 0, 0])
    }])
        
    env = ProfilingEnv(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()

    apple = env.scene.object_registry("name", "apple")
    table = env.scene.object_registry("name", "table")
    knife = env.scene.object_registry("name", "knife")
    knife.keep_still()
    knife.set_position_orientation(
        position=apple.get_position() + np.array([-0.15, 0.0, 0.2]),
        orientation=T.euler2quat([-np.pi / 2, 0, 0]),
    )
    if args.fluids:
        table.states[Covered].set_value(get_system("stain"), True)  # TODO: water is buggy for now, temporarily use stain instead

    output, results = [], []
    for i in range(500):
        if args.robot:
            result = env.step(np.random.uniform(-0.3, 0.3, env.robots[0].action_dim))[4][:4]
        else:
            result = env.step(None)[4][:4]
        results.append(result)

    results = np.array(results)
    for i, title in enumerate(PROFILING_FIELDS):
        field = f"{args.scene}" if args.scene else "Empty scene"
        if args.robot:
            field += ", with Fetch"
        if args.softbody:
            field += ", cloth" 
        if args.fluids:
            field += ", fluids"
        if args.macro_particle_system:
            field += ", macro particles"
        if args.flatcache:
            field += ", flatcache on"
        output.append({
            "name": field,
            "unit": "fps",
            "value": np.mean(results[-300:, i]) * 1000,
            "extra": [title, title]
        })

    ret = []
    if os.path.exists("output.json"):
        with open("output.json", "r") as f:
            ret = json.load(f)
    ret.extend(output)
    with open("output.json", "w") as f:
        json.dump(ret, f)
    og.shutdown()

if __name__ == "__main__":
    main()
