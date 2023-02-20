import os
import time
import logging
import itertools
from typing import List
from collections import OrderedDict
from abc import ABC

import numpy as np
from scipy.spatial.transform import Rotation as R
from OpenGL.GL import *
from pxr import Gf

from omnigibson import assets_path
from omnigibson.macros import macros
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot, GraspingPoint
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.objects.usd_object import USDObject
import omnigibson.utils.transform_utils as T
from omnigibson.utils.vr_utils import calc_z_rot_from_right
from omni.isaac.core.utils.prims import get_prim_at_path

COMPONENT_SUFFIXES = macros.prims.joint_prim.COMPONENT_SUFFIXES
COMPONENT_SUFFIXES = ['x', 'y', 'z', 'rx', 'ry', 'rz']

# Helps eliminate effect of numerical error on distance threshold calculations, especially when part is at the threshold
THRESHOLD_EPSILON = 0.001

NECK_BASE_REL_POS_UNTRACKED = [-0.15, 0, 0.3]
RIGHT_SHOULDER_REL_POS_UNTRACKED = [-0.15, -0.15, 0.3]
LEFT_SHOULDER_REL_POS_UNTRACKED = [-0.15, 0.15, 0.3]
EYE_LOC_POSE_UNTRACKED = ([0.05, 0, 0], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_UNTRACKED = ([0.1, -0.12, -0.4], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_UNTRACKED = ([0.1, 0.12, -0.4], [0.7, 0.7, 0.0, 0.15])

NECK_BASE_REL_POS_TRACKED = [-0.15, 0, -0.15]
RIGHT_SHOULDER_REL_POS_TRACKED = [-0.15, -0.15, -0.15]
LEFT_SHOULDER_REL_POS_TRACKED = [-0.15, 0.15, -0.15]
EYE_LOC_POSE_TRACKED = ([0.05, 0, 0.4], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_TRACKED = ([0.1, -0.15, 0.05], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_TRACKED = ([0.1, 0.15, 0.05], [0.7, 0.7, 0.0, 0.15])

# Body parameters
BODY_HEIGHT_RANGE = (0, 2)  # meters. The body is allowed to clip the floor by about a half.
BODY_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
BODY_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
BODY_MASS = 15  # body mass in kg
BODY_MOVING_FORCE = BODY_MASS * 500

# Hand parameters
HAND_BASE_ROTS = {"right": T.euler2quat([0, 160, -80]), "left": T.euler2quat([0, 160, 80])}
HAND_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HAND_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HAND_DISTANCE_THRESHOLD = 1.2  # distance threshold in meters
HAND_GHOST_HAND_APPEAR_THRESHOLD = 0.15
HAND_OPEN_POSITION = 0
FINGER_CLOSE_POSITION = 1.2
THUMB_CLOSE_POSITION = 0.6
HAND_FRICTION = 2.5
HAND_CLOSE_FORCE = 3
RELEASE_WINDOW = 1 / 30.0  # release window in seconds
THUMB_2_POS = [0, -0.02, -0.05]
THUMB_1_POS = [0, -0.015, -0.02]
PALM_CENTER_POS = [0, -0.04, 0.01]
PALM_BASE_POS = [0, 0, 0.015]
FINGER_TIP_POS = [0, -0.025, -0.055]
HAND_LIFTING_FORCE = 300

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
TRIGGER_FRACTION_THRESHOLD = 0.5
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
ATTACHMENT_BUTTON_TIME_THRESHOLD = 1

# Hand link index constants
PALM_LINK_NAME = "palm"
FINGER_MID_LINK_NAMES = ("Tproximal", "Iproximal", "Mproximal", "Rproximal", "Pproximal")
FINGER_TIP_LINK_NAMES = ("Tmiddle", "Imiddle", "Mmiddle", "Rmiddle", "Pmiddle")
THUMB_LINK_NAME = "Tmiddle"

# Gripper parameters
GRIPPER_GHOST_HAND_APPEAR_THRESHOLD = 0.25
GRIPPER_JOINT_POSITIONS = [0.550569, 0.000000, 0.549657, 0.000000]

# Head parameters
HEAD_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HEAD_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HEAD_DISTANCE_THRESHOLD = 0.5  # distance threshold in meters


class BehaviorRobot(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    A humanoid robot that can be used in VR as an avatar. It has two hands, a body and a head with two cameras.
    """

    def __init__(
            self,
            # Shared kwargs in hierarchy
            prim_path,
            name=None,
            class_id=None,
            uuid=None,
            scale=None,
            visible=True,
            fixed_base=False,
            visual_only=False,
            self_collisions=False,
            load_config=None,

            # Unique to USDObject hierarchy
            abilities=None,

            # Unique to ControllableObject hierarchy
            control_freq=None,
            controller_config=None,
            action_type="continuous",
            action_normalize=False,
            reset_joint_pos=None,

            # Unique to BaseRobot
            obs_modalities="rgb",
            proprio_obs="default",

            # Unique to ManipulationRobot
            grasping_mode="physical",

            # unique to BehaviorRobot
            agent_id=1,
            use_body=True,
            use_ghost_hands=True,

            **kwargs
    ):
        """
        Initializes BehaviorRobot
        Args:
            agent_id (int): unique id of the agent - used in multi-user VR
            image_width (int): width of each camera
            image_height (int): height of each camera
            use_body (bool): whether to use body
            use_ghost_hands (bool): whether to use ghost hand
        """

        super(BehaviorRobot, self).__init__(
            prim_path=prim_path,
            name=name,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            grasping_mode=grasping_mode,
            **kwargs,
        )

        # Basic parameters
        self.agent_id = agent_id
        self.use_body = use_body
        self.use_ghost_hands = use_ghost_hands
        self.simulator = None
        self._world_base_fixed_joint_prim = None

        # Activation parameters
        self.first_frame = True
        # whether hand or body is in contact with other objects (we need this since checking contact list is costly)
        self.part_is_in_contact = {hand_name: False for hand_name in self.arm_names + ["body"]}

        # Whether the VR system is actively hooked up to the VR agent.
        self.vr_attached = False
        self._vr_attachment_button_press_timestamp = None
        self._most_recent_trigger_fraction = {}

        # setup eef parts
        self.parts = OrderedDict()
        self.parts["lh"] = BRPart(
            name="lh", parent=self, prim_path="lh_base", eef_type="hand",
            rel_offset=LEFT_SHOULDER_REL_POS_UNTRACKED, **kwargs
        )

        self.parts["rh"] = BRPart(
            name="rh", parent=self,  prim_path="rh_base", eef_type="hand",
            rel_offset=RIGHT_SHOULDER_REL_POS_UNTRACKED, **kwargs
        )

        self.parts["head"] = BRPart(
            name="head", parent=self,  prim_path="eye", eef_type="head",
            rel_offset=NECK_BASE_REL_POS_UNTRACKED, **kwargs
        )

    @property
    def usd_path(self):
        return os.path.join(assets_path, "models/vr_agent/usd/BehaviorRobot.usd")

    @property
    def model_name(self):
        return "BehaviorRobot"
    
    @property
    def n_arms(self):
        return 2
    
    @property
    def arm_names(self):
        return ["lh", "rh"]
    
    @property
    def eef_link_names(self):
        dic = {arm: f"{arm}_palm" for arm in self.arm_names}
        dic["head"] = "eye"
        return dic

    @property
    def arm_link_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {arm: [f"{arm}_{component}" for component in COMPONENT_SUFFIXES] for arm in self.arm_names + ['head']}
    
    @property
    def finger_link_names(self):
        return {
            arm: [
                "%s_%s" % (arm, link_name)
                for link_name in itertools.chain(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)
            ]
            for arm in self.arm_names
        }
    
    @property
    def base_joint_names(self):
        return [f"base_{component}_joint" for component in COMPONENT_SUFFIXES]
    
    @property
    def arm_joint_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {eef: [f"{eef}_{component}_joint" for component in COMPONENT_SUFFIXES] for eef in self.arm_names + ["head"]}
    
    @property
    def finger_joint_names(self):
        return {
            arm: (
                # palm-to-proximal joints.
                ["%s_%s__%s_palm" % (arm, to_link, arm) for to_link in FINGER_MID_LINK_NAMES]
                +
                # proximal-to-tip joints.
                [
                    "%s_%s__%s_%s" % (arm, to_link, arm, from_link)
                    for from_link, to_link in zip(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)
                ]
            )
            for arm in self.arm_names
        }
    
    @property
    def base_control_idx(self):
        # TODO: might need to refactor out joints
        joints = list(self.joints.keys())
        return tuple(joints.index(joint) for joint in self.base_joint_names)
    
    @property
    def arm_control_idx(self):
        joints = list(self.joints.keys())
        return {
            arm: [joints.index(f"{arm}_{component}_joint") for component in COMPONENT_SUFFIXES]
            for arm in self.arm_names
        }

    @property
    def gripper_control_idx(self):
        joints = list(self.joints.values())
        return {arm: [joints.index(joint) for joint in arm_joints] for arm, arm_joints in self.finger_joints.items()}

    @property
    def camera_control_idx(self):
        joints = list(self.joints.keys())
        return [joints.index(f"head_{component}_joint") for component in COMPONENT_SUFFIXES]

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def controller_order(self):
        controllers = ["base", "camera"]
        for arm_name in self.arm_names:
            controllers += [f"arm_{arm_name}", f"gripper_{arm_name}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = {
            "base": "JointController",
            "camera": "JointController"
        }
        controllers.update({f"arm_{arm_name}": "JointController" for arm_name in self.arm_names})
        controllers.update({f"gripper_{arm_name}": "MultiFingerGripperController" for arm_name in self.arm_names})
        return controllers

    @property
    def _default_base_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "motor_type": "position",
            "dof_idx": self.base_control_idx,
            "command_input_limits": None,
            "command_output_limits": None,
        }

    @property
    def _default_arm_joint_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_input_limits": None,
                "command_output_limits": None,  # TODO: Set this to "default" after control limits are fixed.
                "use_delta_commands": False,
            }
        return dic
    
    @property
    def _default_gripper_multi_finger_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "inverted": True,
                "mode": "smooth",
            }
        return dic
    
    @property
    def _default_camera_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "command_input_limits": None,
            "command_output_limits": None,
            "use_delta_commands": False,
        }

    @property
    def _default_controller_config(self):
        controllers = {
            "base": {"JointController": self._default_base_joint_controller_config},
            "camera": {"JointController": self._default_camera_joint_controller_config},
        }
        controllers.update(
            {
                f"arm_{arm_name}": {"JointController": self._default_arm_joint_controller_configs[arm_name]}
                for arm_name in self.arm_names
            }
        )
        controllers.update(
            {
                f"gripper_{arm_name}": {
                    "MultiFingerGripperController": self._default_gripper_multi_finger_controller_configs[arm_name],
                }
                for arm_name in self.arm_names
            }
        )
        return controllers

    def load(self, simulator=None):
        prim = super(BehaviorRobot, self).load(simulator)
        for part in self.parts.values():
            part.load(simulator)
        return prim

    def _post_load(self):
        super()._post_load()
        self._world_base_fixed_joint_prim = get_prim_at_path(os.path.join(self.root_link.prim_path, "world_to_base"))
        position, orientation = self.get_position_orientation()
        # Set the world-to-base fixed joint to be at the robot's current pose
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
        self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*orientation[[3, 0, 1, 2]]))

    def _create_discrete_action_space(self):
        raise ValueError("BehaviorRobot does not support discrete actions!")

    def test(self):
        # super()._initialize()
        # set joint mass and rigid body properties
        for link in self.links.values():
            link.mass = 0.1
            link.ccd_enabled = True
        for arm in self.arm_names:
            for link in self.finger_link_names[arm]:
                self.links[link].mass = 0.001
                self.links[link].ccd_enabled = True
        self.links["base"].mass = 30

        # set base joint properties
        for joint_name in self.base_joint_names:
            self.joints[joint_name].friction = 2.5
            self.joints[joint_name].max_effort = 1e3
            self.joints[joint_name].stiffness = 1e9
            self.joints[joint_name].damping = 10

        # set arm joint properties
        for arm in self.arm_joint_names:
            for joint_name in self.arm_joint_names[arm]:
                self.joints[joint_name].friction = 2.5
                self.joints[joint_name].max_effort = 1e3
                self.joints[joint_name].stiffness = 1e9
                self.joints[joint_name].damping = 10
        # set finger joint properties
        for arm in self.finger_joint_names:
            for joint_name in self.finger_joint_names[arm]:
                self.joints[joint_name].max_effort = 25
                self.joints[joint_name].stiffness = 1e9
        # set vision sensor properties
        for sensor in self._sensors.values():
            sensor.clipping_range = (0.01, 1e5)

    @property
    def base_footprint_link_name(self):
        """
        Name of the actual root link that we are interested in. 
        """
        return "body"

    @property
    def base_footprint_link(self):
        """
        Returns:
            RigidPrim: base footprint link of this object prim
        """
        return self._links[self.base_footprint_link_name]


    def get_position_orientation(self):
        # If the simulator is playing, return the pose of the base_footprint link frame
        if self._dc is not None and self._dc.is_simulating():
            return self.base_footprint_link.get_position_orientation()

        # Else, return the pose of the robot frame
        else:
            return super().get_position_orientation()

    def set_position_orientation(self, position=None, orientation=None):
        super().set_position_orientation(position, orientation)
        # Move the joint frame for the world_base_joint
        if self._world_base_fixed_joint_prim is not None:
            if position is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
            if orientation is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*np.float_(orientation)[[3, 0, 1, 2]]))

    @property
    def assisted_grasp_start_points(self):
        side_coefficients = {"lh": np.array([1, -1, 1]), "rh": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_BASE_POS),
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_CENTER_POS * side_coefficients[arm]),
                GraspingPoint(
                    link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_1_POS * side_coefficients[arm]
                ),
                GraspingPoint(
                    link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_2_POS * side_coefficients[arm]
                ),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        side_coefficients = {"lh": np.array([1, -1, 1]), "rh": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s_%s" % (arm, finger), position=FINGER_TIP_POS * side_coefficients[arm])
                for finger in FINGER_TIP_LINK_NAMES
            ]
            for arm in self.arm_names
        }

    def update_hand_contact_info(self):
        self.part_is_in_contact["body"] = len(self.links["body"].contact_list())
        for hand_name in self.arm_names:
            self.part_is_in_contact[hand_name] = len(self.eef_links[hand_name].contact_list()) \
               or np.any([len(finger.contact_list()) for finger in self.finger_links[hand_name]])

    def update_vr_render(self):
        obs = self.get_obs()
        glBindTexture(GL_TEXTURE_2D, self._simulator.vr_text_left_id)
        obs_left_eye = obs['robot:eye_left_eye_rgb']
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_width, self.image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, obs_left_eye)
        glBindTexture(GL_TEXTURE_2D, self._simulator.vr_text_right_id)
        obs_right_eye = obs['robot:eye_right_eye_rgb']
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_width, self.image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, obs_right_eye)

    def gen_vr_robot_action(self):
        """
        Generates an action for the BehaviorRobot to perform based on VrData collected this frame.

        Action space (all non-normalized values that will be clipped if they are too large)
        * See BehaviorRobot.py for details on the clipping thresholds for
        Body:
        - 6DOF pose delta - relative to body frame from previous frame
        Eye:
        - 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
        Left hand, right hand (in that order):
        - 6DOF pose delta - relative to body frame (same as above)
        - Trigger fraction

        Total size: 26
        """
        # Actions are stored as 1D numpy array
        action = np.zeros(26)
        action[[18, 25]] = 1 # trigger frac defaults to 1 (hands open)
        # Get VrData for the current frame
        v = self._simulator.gen_vr_data()

        # If a double button press is recognized for ATTACHMENT_BUTTON_TIME_THRESHOLD seconds, attach/detach the VR
        # system as needed. Forward a zero action to the robot if deactivated or if a switch is recognized.
        attach_or_detach = self._simulator.get_action_button_state(
            "left_controller", "reset_agent", v
        ) and self._simulator.get_action_button_state("right_controller", "reset_agent", v)
        if attach_or_detach:
            # If the button just recently started being pressed, record the time.
            if self._vr_attachment_button_press_timestamp is None:
                logging.info("Double button press detected.")
                self._vr_attachment_button_press_timestamp = time.time()

            # If the button has been pressed for ATTACHMENT_BUTTON_TIME_THRESHOLD seconds, attach/detach.
            if time.time() - self._vr_attachment_button_press_timestamp > ATTACHMENT_BUTTON_TIME_THRESHOLD:
                # Replace timestamp with infinity so that the condition won't re-trigger until button is released.
                self._vr_attachment_button_press_timestamp = float("inf")

                # Flip the attachment state.
                self.vr_attached = not self.vr_attached
                logging.info("VR kit {} BehaviorRobot.".format("attached to" if self.vr_attached else "detached from"))

                # We don't want to fill in an action in this case.
                return action
        else:
            # If the button is released, stop keeping track.
            self._vr_attachment_button_press_timestamp = None

        # If the VR system is not attached to the robot, return a zero action.
        if not self.vr_attached:
            return action

        # otherwise, generate action space from controller inputs and device positions

        # Update body action space
        hmd_is_valid, hmd_pos, hmd_orn, hmd_r = v.query("hmd")[:4]
        prev_body_pos, prev_body_orn = self.get_position_orientation()

        if self._simulator.vr_settings.using_tracked_body:
            torso_is_valid, torso_pos, torso_orn = v.query("torso_tracker")
            if torso_is_valid:
                des_body_pos, des_body_orn = torso_pos, torso_orn
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn
        else:
            if hmd_is_valid:
                des_body_pos, des_body_orn = hmd_pos.copy(), T.euler2quat([0, 0, calc_z_rot_from_right(hmd_r)])
                des_body_pos[2] -= 0.45
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn

        root_pos, root_orn = self.root_link.get_position_orientation()
        new_body_pos, new_body_orn = T.relative_pose_transform(des_body_pos, des_body_orn, root_pos, root_orn)
        new_body_rpy = T.quat2euler(new_body_orn)
        action[self.controller_action_idx["base"]] = np.concatenate([new_body_pos, new_body_rpy])

        # Update action space for other VR objects
        for part_name, eef_part in self.parts.items():
            # Process local transform adjustments
            prev_local_pos, prev_local_orn = eef_part.local_position_orientation
            if part_name == "head":
                valid, world_pos, world_orn = hmd_is_valid, hmd_pos, hmd_orn
            else:
                controller_name = "left_controller" if part_name == "lh" else "right_controller"
                valid, world_pos, _ = v.query(controller_name)[:3]
                # Need rotation of the model so that it will appear aligned with the physical controller in VR
                world_orn = v.query(controller_name)[6]
            # Keep in same world position as last frame if controller/tracker data is not valid
            if not valid:
                world_pos, world_orn = eef_part.world_position_orientation
            # Get desired local position and orientation transforms
            des_local_pos, des_local_orn = T.relative_pose_transform(
                world_pos, world_orn, root_pos, root_orn
            )

            # generate actions for this eef part
            cur_eef_actions = eef_part.gen_eef_actions(des_local_pos, des_local_orn)

            controller_name = "camera" if part_name == "head" else "arm_" + part_name
            action[self.controller_action_idx[controller_name]] = cur_eef_actions
            # Process trigger fraction and reset for controllers
            if part_name in self.arm_names:
                if valid:
                    button_name = "left_controller_button" if part_name == "lh" else "right_controller_button"
                    trig_frac = v.query(button_name)[1]
                    self._most_recent_trigger_fraction[part_name] = trig_frac
                else:
                    # Use the last trigger fraction if no valid input was received from controller.
                    trig_frac = (
                        self._most_recent_trigger_fraction[part_name]
                        if part_name in self._most_recent_trigger_fraction
                        else 0.0
                    )

                grip_controller_name = "gripper_" + part_name
                scaled_trig_frac = -2 * trig_frac + 1  # Map from (0, 1) to (-1, 1) range, inverted.
                action[self.controller_action_idx[grip_controller_name]] = scaled_trig_frac

                # If we reset, action is 1, otherwise 0
                reset_action = v.query("reset_actions")[0] if part_name == "left" else v.query("reset_actions")[1]
                if reset_action:
                    self.parts[part_name].set_position_orientation(world_pos, T.quat2euler(world_orn))
                # update ghost hand if necessary
                if self.use_ghost_hands and part_name is not "head":
                    self.parts[part_name].update_ghost_hands(world_pos, world_orn)
        return action
    

class BRPart(ABC):
    """This is the interface that all BehaviorRobot eef parts must implement."""

    def __init__(self, name: str, parent: BehaviorRobot, prim_path: str, eef_type: str, rel_offset: List[float]):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        Args:
            name (str): unique name of this BR part
            parent (BehaviorRobot): the parent BR object
            prim_path (str): prim path to the root link of the eef
            eef_type (str): type of eef. One of hand, head
            rel_offset (List[float]): relative position offset to the shoulder prim
        """
        self.name = name
        self.parent = parent
        self.prim_path = prim_path
        self.eef_type = eef_type

        self.ghost_hand = None
        self._root_link = None
        self._world_position_orientation = ([0, 0, 0], [0, 0, 0, 1])
        self._body_pose_in_shoulder_frame = [-i for i in rel_offset]

    def load(self, simulator):
        self._root_link = self.parent.links[self.prim_path]
        # setup ghost hand
        if self.eef_type == "hand":
            gh_name = f"ghost_hand_{self.name}"
            self.ghost_hand = USDObject(
                prim_path=f"/World/{gh_name}",
                usd_path=os.path.join(assets_path, f"models/vr_agent/usd/{gh_name}.usd"),
                name=gh_name,
                scale=0.001,
                visible=False,
                visual_only=True,
            )
            simulator.import_object(self.ghost_hand)

    @property
    def local_position_orientation(self):
        """
        Get local position and orientation w.r.t. to the body
        Return:
            Tuple[Array[x, y, z], Array[x, y, z, w]]

        """
        return T.relative_pose_transform(*self.world_position_orientation, *self.parent.get_position_orientation())

    @property
    def world_position_orientation(self):
        """
        Get position and orientation in the world space
        Return:
            Tuple[Array[x, y, z], Array[x, y, z, w]]
        """
        return self._root_link.get_position_orientation()

    def set_position_orientation(self, pos: List[float], orn: List[float]):
        """
        Call back function to set the base's position
        """
        self.parent.joints[f"{self.name}_x_joint"].set_pos(pos[0], target=False)
        self.parent.joints[f"{self.name}_y_joint"].set_pos(pos[1], target=False)
        self.parent.joints[f"{self.name}_z_joint"].set_pos(pos[2], target=False)
        self.parent.joints[f"{self.name}_rx_joint"].set_pos(orn[0], target=False)
        self.parent.joints[f"{self.name}_ry_joint"].set_pos(orn[1], target=False)
        self.parent.joints[f"{self.name}_rz_joint"].set_pos(orn[2], target=False)


    def gen_eef_actions(self, pos: List[float], orn: List[float]):
        """
        generate 7-DOF delta action for the eef based on its delta positions and rotations

        Args:
            pos (List[float]): list of positions [x, y, z]
            orn (List[float]): list of rotations [x, y, z, w]

        Return:
            List[float]: List of 6 actions
        """
        eef_actions = np.zeros(6)
        target_in_shoulder_frame_pos, target_in_shoulder_frame_orn = T.pose_transform(
            self._body_pose_in_shoulder_frame, [0, 0, 0, 1], pos, orn
        )
        eef_actions[:3] = target_in_shoulder_frame_pos
        eef_actions[3:] = R.from_quat(target_in_shoulder_frame_orn).as_euler("XYZ")
        return eef_actions

    def update_ghost_hands(self, pos, orn):
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        Args:
            pos (List[float]): list of positions [x, y, z]
            orn (List[float]): list of rotations [x, y, z, w]
        """
        assert self.eef_type == "hand", "ghost hand is only valid for BR hand!"
        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position(pos)
        self.ghost_hand.set_orientation(orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(pos - self.world_position_orientation[0])
        should_visible = dist_to_real_controller > HAND_GHOST_HAND_APPEAR_THRESHOLD

        # Only toggle visibility if we are transition from hidden to unhidden, or the other way around
        if self.ghost_hand.visible is not should_visible:
            self.ghost_hand.visible = should_visible
