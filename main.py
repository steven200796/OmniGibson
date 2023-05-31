import os

import yaml

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Load the config
    config_filename = "main_config.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "coffee_table"]


    # Load the specs of the object categories, e.g., common scaling factor
    # avg_category_spec = get_og_avg_category_specs()

    # obj_cfg = dict(
    #     type="USDObject",
    #     name="obj",
    #     usd_path=r"C:\Users\icaro\Downloads\Kaiser_Bread_Roll.usd",
    #     # category="baguette",
    #     visual_only=False,
    #     scale=[1.0, 1.0, 1.0],
    #     position=[0, 0, 1.0],
    # )

    # config["objects"] = [obj_cfg]
    # Load the environment
    env = og.Environment(configs=config)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    scene = env.scene
    robot = env.robots[0]
    robot.set_position([-0.5, 2.4, 0.0])
    controller = StarterSemanticActionPrimitives(None, scene, robot)
    # obj = scene
    navigate_controller = controller._navigate_to_obj(obj)
    
    for i in range(10000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            og.log.info("Episode finished after {} timesteps".format(i + 1))
            break
    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
