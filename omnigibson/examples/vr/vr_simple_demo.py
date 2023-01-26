import numpy as np
import omnigibson as og
from omnigibson.scenes.scene_base import Scene
from omnigibson.robots.behavior_robot import BehaviorRobot
from omnigibson.objects.usd_object import USDObject


og.sim.set_vr_start_pos(start_pos=[1, 0, 1.5])
# Load scene
scene = Scene()
# Import scene
og.sim.initialize_vr()
og.sim.import_scene(scene=scene)

banana_scale = 0.9
#import relavant object
table = USDObject(
    prim_path='/World/table', usd_path=f"{og.og_dataset_path}/objects/desk/19898/usd/19898.usd", name="table", fixed_base=True,
)
obj1 = USDObject(
    prim_path='/World/obj1', usd_path=f"{og.og_dataset_path}/objects/banana/09_0/usd/09_0.usd", name="obj1", scale=banana_scale
)
obj2 = USDObject(
    prim_path='/World/obj2', usd_path=f"{og.og_dataset_path}/objects/banana/09_0/usd/09_0.usd", name="obj2", scale=banana_scale
)
obj3 = USDObject(
    prim_path='/World/obj3', usd_path=f"{og.og_dataset_path}/objects/banana/09_0/usd/09_0.usd", name="obj3", scale=banana_scale
)
obj4 = USDObject(
    prim_path='/World/obj4', usd_path=f"{og.og_dataset_path}/objects/banana/09_0/usd/09_0.usd", name="obj4", scale=banana_scale
)
og.sim.import_object(obj=table)
og.sim.import_object(obj=obj1)
og.sim.import_object(obj=obj2)
og.sim.import_object(obj=obj3)
og.sim.import_object(obj=obj4)
table.set_position_orientation(np.array([2, 0, 1]), np.array([0, 0, -0.7071068, 0.7071068]))
obj1.set_position([2., 0, 2])
obj2.set_position([2., -0.1, 2.02])
obj3.set_position([2., 0, 2.04])
obj4.set_position([2, 0.1, 2.06])
for link in table.links.values():
    link.ccd_enabled = True
for link in obj1.links.values():
    link.ccd_enabled = True
for link in obj2.links.values():
    link.ccd_enabled = True
for link in obj3.links.values():
    link.ccd_enabled = True
for link in obj4.links.values():
    link.ccd_enabled = True
# Import robot
robot = BehaviorRobot(prim_path=f"/World/robot", name="robot", image_height=1280, image_width=1280)
og.sim.import_object(obj=robot)
og.sim.register_main_vr_robot(robot)
# og.sim.set_vr_start_pos([0, 0, 2])

# The simulator must always be playing and have a single step taken in order to initialize any added objects
og.sim.play()
og.sim.step()
robot.set_position([2., 0, 2])


# Set the camera to a well-known, good position for viewing the robot
og.sim.viewer_camera.set_position_orientation(
    position=np.array([-0.300727, -3.7592, 2.03752]),
    orientation=np.array([0.53647195, -0.02424788, -0.03808954, 0.84270936]),
)

# We also enable keyboard teleoperation of the simulator's viewer camera for convenience
og.sim.enable_viewer_camera_teleoperation()

og.sim.viewer_camera.remove()

for i in range(1000000):
    if og.sim.is_playing():
        robot.apply_action(robot.gen_vr_robot_action())
    og.sim.step(print_stats=True)

og.app.close()
