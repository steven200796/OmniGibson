import numpy as np
import omnigibson as og
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.robots.behavior_robot import BehaviorRobot

##### SET THIS ######
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{og.og_dataset_path}/scenes/{SCENE_ID}/usd/{SCENE_ID}_best_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

# Load scene
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
og.sim.initialize_vr()
og.sim.import_scene(scene=scene)

# Import robot
robot = BehaviorRobot(prim_path=f"/World/robot", name="robot", image_height=2000, image_width=2000)
og.sim.import_object(obj=robot)
og.sim.register_main_vr_robot(robot)

# The simulator must always be playing and have a single step taken in order to initialize any added objects
og.sim.play()
og.sim.step()

# Set the camera to a well-known, good position for viewing the robot
og.sim.viewer_camera.set_position_orientation(
    position=np.array([-0.300727, -3.7592,  2.03752]),
    orientation=np.array([0.53647195, -0.02424788, -0.03808954, 0.84270936]),
)

# We also enable keyboard teleoperation of the simulator's viewer camera for convenience
og.sim.enable_viewer_camera_teleoperation()

# og.sim.viewer_camera.remove()

# toggle = 1
for i in range(1000000):
    robot.apply_action(robot.gen_vr_robot_action())
    og.sim.step(print_stats=True)

og.app.close()
