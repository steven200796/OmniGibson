import time
import numpy as np

from time import sleep
from typing import List, Optional
from OpenGL.GL import *
from omnigibson.utils.vr_utils import VrSettings, VrHUDOverlay, VrStaticImageOverlay
from omnigibson.vr.VRSys import VRSys
from omnigibson.robots.manipulation_robot import IsGraspingState
from omnigibson.robots.behavior_robot import HAND_BASE_ROTS
from omnigibson.simulator import Simulator
from omnigibson.utils.vr_utils import VR_CONTROLLERS, VR_DEVICES, VrData, calc_offset
import omnigibson.utils.transform_utils as T


class SimulatorVR(Simulator):
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
            self,
            gravity=9.81,
            physics_dt=1 / 120.0,
            rendering_dt=1 / 60.0,
            stage_units_in_meters: float = 1.0,
            viewer_width=1600,
            viewer_height=1440,
            device_idx=None,
    ):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param solver_iterations: number of solver iterations to feed into pybullet, can be reduced to increase speed.
            pybullet default value is 50.
        :param use_variable_step_num: whether to use a fixed (1) or variable physics step number
        :param mode: choose mode from headless, headless_tensor, gui_interactive, gui_non_interactive
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param rendering_settings: settings to use for mesh renderer
        :param vr_settings: settings to use for VR in simulator and MeshRendererVR
        """
        # Starting position for the VR (default set to None if no starting position is specified by the user)
        self.vr_settings = VrSettings()
        self.vr_overlay_initialized = False
        self.vr_start_position = None
        self._vr_attachment_button_press_timestamp = float("inf")

        # Duration of a vsync frame - assumes 90Hz refresh rate
        self.vsync_frame_dur = 11.11e-3
        # Timing variables for functions called outside of step() that also take up frame time
        self.frame_end_time = None

        # Variables for data saving and replay in VR
        self.last_physics_step = -1
        self.last_render_timestep = -1
        self.last_physics_step_num = -1
        self.last_frame_dur = -1
        self.frame_count = 0

        super().__init__(
            gravity,
            physics_dt,
            rendering_dt,
            stage_units_in_meters,
            viewer_width,
            viewer_height,
            device_idx,
        )

        # Get expected number of vsync frames per iGibson frame Note: currently assumes a 90Hz VR system
        self.vsync_frame_num = int(round(rendering_dt / self.vsync_frame_dur))

        # Total amount of time we want non-blocking actions to take each frame
        # Leave a small amount of time before the last vsync, just in case we overrun
        self.non_block_frame_time = (self.vsync_frame_num - 1) * self.vsync_frame_dur + (
            5e-3 if self.vr_settings.curr_device == "OCULUS" else 10e-3
        )
        self.rendering_dt = rendering_dt
        self.main_vr_robot = None
        self.vr_hud = None
        self.vr_text_left_id, self.vr_text_right_id = None, None
        self.vr_sys = None

    def __del__(self):
        if self.vr_sys:
            self.vr_sys.release()
        return super().__del__()

    def initialize_vr(self):
        self.vr_sys = VRSys()
        self.vr_sys.init(self.vr_sys.hasEyeTrackingSupport())
        self.initialize_vr_render()

    def add_vr_overlay_text(
        self,
        text_data="PLACEHOLDER: PLEASE REPLACE!",
        font_name="OpenSans",
        font_style="Regular",
        font_size=48,
        color=[0, 0, 0],
        pos=[20, 80],
        size=[70, 80],
        scale=1.0,
        background_color=[1, 1, 1, 0.8],
    ):
        """
        Creates Text for use in a VR overlay. Returns the text object to the caller,
        so various settings can be changed - eg. text content, position, scale, etc.
        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_style: style of font - one of [regular, italic, bold]
        :param font_size: size of font to render
        :param color: [r, g, b] color
        :param pos: [x, y] position of top-left corner of text box, in percentage across screen
        :param size: [w, h] size of text box in percentage across screen-space axes
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - default is semi-transparent white so text is easy to read in VR
        """
        if not self.vr_overlay_initialized:
            # This function automatically creates a VR text overlay the first time text is added
            self.gen_vr_hud()
            self.vr_overlay_initialized = True

        # Note: For pos/size - (0,0) is bottom-left and (100, 100) is top-right
        # Calculate pixel positions for text
        pixel_pos = [int(pos[0] / 100.0 * self.viewer_width), int(pos[1] / 100.0 * self.viewer_height)]
        pixel_size = [int(size[0] / 100.0 * self.viewer_width), int(size[1] / 100.0 * self.viewer_height)]
        return self.renderer.add_text(
            text_data=text_data,
            font_name=font_name,
            font_style=font_style,
            font_size=font_size,
            color=color,
            pixel_pos=pixel_pos,
            pixel_size=pixel_size,
            scale=scale,
            background_color=background_color,
            render_to_tex=True,
        )

    def gen_vr_hud(self):
        """
        Generates VR HUD (heads-up-display).
        """
        # Create a unique overlay name based on current nanosecond
        uniq_name = "overlay{}".format(time.perf_counter())
        self.vr_hud = VrHUDOverlay(uniq_name, self, width=self.vr_settings.hud_width, pos=self.vr_settings.hud_pos)
        self.vr_hud.set_overlay_show_state(True)

    def add_overlay_image(self, image_fpath, width=1, pos=[0, 0, -1]):
        """
        Add an image with a given file path to the VR overlay. This image will be displayed
        in addition to any text that the users wishes to display. This function returns a handle
        to the VrStaticImageOverlay, so the user can display/hide it at will.
        """
        uniq_name = "overlay{}".format(time.perf_counter())
        static_overlay = VrStaticImageOverlay(uniq_name, self, image_fpath, width=width, pos=pos)
        static_overlay.set_overlay_show_state(True)
        return static_overlay

    def set_hud_show_state(self, show_state: bool):
        """
        Shows/hides the main VR HUD.
        Args:
            show_state (bool): whether to show HUD or not
        """
        self.vr_hud.set_overlay_show_state(show_state)

    def get_hud_show_state(self):
        """
        Returns the show state of the main VR HUD.
        """
        return self.vr_hud.get_overlay_show_state()

    def step_vr_system(self):
        # Update VR compositor and VR data
        vr_system_start = time.perf_counter()
        # Note: this should only be called once per frame - use get_vr_events to read the event data list in
        # subsequent read operations
        self.poll_vr_events()
        # This is necessary to fix the eye tracking data for the current frame, since it is multi-threaded
        self.fix_eye_tracking_data()
        # Move user to their starting location
        self.perform_vr_start_pos_move()
        # Update VR data and wait until 3ms before the next vsync
        self.vr_sys.pollVRPosesAndStates()
        # Update VR system data - eg. offsets, haptics, etc.
        self.vr_system_update()
        vr_system_dur = time.perf_counter() - vr_system_start
        return vr_system_dur

    def step(self, render=True, force_playing=False, print_stats=False):
        """
        Step the simulation when using VR. Order of function calls:
        1) Simulate physics
        2) Render frame
        3) Submit rendered frame to VR compositor
        4) Update VR data for use in the next frame
        """
        assert (
            self.scene is not None
        ), "A scene must be imported before running the simulator. Use EmptyScene for an empty scene."

        # Calculate time outside of step
        outside_step_dur = 0
        if self.frame_end_time is not None:
            outside_step_dur = time.perf_counter() - self.frame_end_time
        # Simulate Physics in OmniGibson
        omni_step_start_time = time.perf_counter()
        super().step(render, force_playing)
        # update collision status of BR hand
        if self.main_vr_robot:
            self.main_vr_robot.update_hand_contact_info()
        omni_dur = time.perf_counter() - omni_step_start_time
        # render to headset
        render_start_time = time.perf_counter()
        self.render_vr()
        render_dur = time.perf_counter() - render_start_time
        # Sleep until last possible Vsync
        pre_sleep_dur = outside_step_dur + omni_dur + render_dur
        sleep_start_time = time.perf_counter()
        if pre_sleep_dur < self.non_block_frame_time:
            sleep(self.non_block_frame_time - pre_sleep_dur)
        sleep_dur = time.perf_counter() - sleep_start_time
        vr_system_dur = self.step_vr_system()

        # Calculate final frame duration
        # Make sure it is non-zero for FPS calculation (set to max of 1000 if so)
        frame_dur = max(1e-3, pre_sleep_dur + sleep_dur + vr_system_dur)

        # Set variables for data saving and replay
        self.last_physics_step = omni_dur
        self.last_render_timestep = render_dur
        self.last_frame_dur = frame_dur

        if print_stats:
            print("Frame number {} statistics (ms)".format(self.frame_count))
            print("Total out-of-step duration: {}".format(outside_step_dur * 1000))
            print("Total omni duration: {}".format(omni_dur * 1000))
            print("Total render duration: {}".format(render_dur * 1000))
            print("Total sleep duration: {}".format(sleep_dur * 1000))
            print("Total VR system duration: {}".format(vr_system_dur * 1000))
            print("Total frame duration: {} and fps: {}".format(frame_dur * 1000, 1 / frame_dur))
            print("-------------------------")

        self.frame_count += 1
        self.frame_end_time = time.perf_counter()

    def render_vr(self):
        """
        Renders VR scenes.
        """
        if self.main_vr_robot:
            self.main_vr_robot.update_vr_render()
        self.vr_sys.render(self.vr_text_left_id, self.vr_text_right_id)

    def vr_system_update(self):
        """
        Updates the VR system for a single frame. This includes moving the vr offset,
        adjusting the user's height based on button input, and triggering haptics.
        """
        # Update VR offset using appropriate controller
        if self.vr_settings.touchpad_movement:
            vr_offset_device = "{}_controller".format(self.vr_settings.movement_controller)
            curr_offset = self.vr_sys.getVROffset()
            is_valid, _, touch_x, touch_y, _ = self.vr_sys.getControllerButtonData(vr_offset_device)
            if is_valid:
                curr_offset = calc_offset(
                    self, curr_offset, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device
                )
            # Adjust user height based on y-axis (vertical direction) touchpad input
            vr_height_device = "left_controller" if self.vr_settings.movement_controller == "right" else "right_controller"
            is_valid, _, _, height_y, _ = self.vr_sys.getControllerButtonData(vr_height_device)
            if is_valid:
                hmd_height = self.vr_sys.getDevicePose("hmd")[-1][2]
                if height_y < -0.7:
                    curr_offset[2] = max(curr_offset[2] - 0.01, self.vr_settings.height_bounds[0] - hmd_height)
                elif height_y > 0.7:
                    curr_offset[2] = min(curr_offset[2] + 0.01, self.vr_settings.height_bounds[1] - hmd_height)
            self.vr_sys.setVROffset(*curr_offset)
        # Update haptics for controllers
        if self.main_vr_robot:
            is_body_in_collision = self.main_vr_robot.part_is_in_contact["body"]
            for controller, hand_name in [("left_controller", "lh"), ("right_controller", "rh")]:
                is_valid, _, _, _ = self.vr_sys.getDevicePose(controller)
                if is_valid:
                    if (
                        self.main_vr_robot.part_is_in_contact[hand_name]
                        or self.main_vr_robot.is_grasping(hand_name) == IsGraspingState.TRUE
                    ):
                        self.vr_sys.triggerHapticPulse(controller, 0.3)
                    elif is_body_in_collision:
                        self.vr_sys.triggerHapticPulse(controller, 0.9)

    def register_main_vr_robot(self, vr_robot):
        """
        Register the robot representing the VR user.
        """
        self.main_vr_robot = vr_robot

    def gen_vr_data(self):
        """
        Generates a VrData object containing all the data required to describe the VR system in the current frame.
        This data is used to power the BehaviorRobot each frame.
        """
        v = dict()
        for device in VR_DEVICES:
            is_valid, trans, rot, _ = self.vr_sys.getDevicePose(device)
            device_data = [is_valid, trans.tolist(), rot.tolist()]
            device_data.extend(self.vr_sys.getDeviceCoordinateSystem(device))
            v[device] = device_data
            if device in VR_CONTROLLERS:
                v[f"{device}_button"] = self.vr_sys.getControllerButtonData(device)

        for hand in ["right", "left"]:
            # Base rotation quaternion
            base_rot = HAND_BASE_ROTS[hand]
            # Raw rotation of controller
            if v[f"{hand}_controller"][0]:
                controller_rot = v[f"{hand}_controller"][2]
            else:
                controller_rot = [0, 0, 0, 1]
            # Use dummy translation to calculation final rotation
            final_rot = T.pose_transform([0, 0, 0], controller_rot, [0, 0, 0], base_rot)[1]
            v[f"{hand}_controller"].append(final_rot)

        is_valid, torso_trans, torso_rot, _ = self.vr_sys.getDevicePose(self.vr_settings.torso_tracker_serial)
        v["torso_tracker"] = [is_valid, torso_trans.tolist(), torso_rot.tolist()]
        v["eye_data"] = self.eye_tracking_data
        v["event_data"] = self.get_vr_events()
        reset_actions = []
        for controller in VR_CONTROLLERS:
            reset_actions.append(self.query_vr_event(controller, "reset_agent"))
        v["reset_actions"] = reset_actions
        return VrData(v)

    def perform_vr_start_pos_move(self):
        """
        Sets the VR position on the first step iteration where the hmd tracking is valid. Not to be confused
        with self.set_vr_start_position, which simply records the desired start position before the simulator starts running.
        """
        # Update VR start position if it is not None and the hmd is valid
        # This will keep checking until we can successfully set the start position
        if self.vr_start_position:
            hmd_is_valid, hmd_position, _, _ = self.vr_sys.getDevicePose("hmd")
            if hmd_is_valid:
                offset_to_start = np.array(self.vr_start_position) - hmd_position
                if self.vr_start_height_offset is not None:
                    offset_to_start[2] = self.vr_start_height_offset
                self.vr_sys.setVROffset(*offset_to_start)
                self.vr_start_position = None

    def fix_eye_tracking_data(self):
        """
        Calculates and fixes eye tracking data to its value during step(). 
        This is necessary, since multiple
        calls to get eye tracking data return different results, due to the SRAnipal multithreaded loop that
        runs in parallel to the iGibson main thread
        """
        self.eye_tracking_data = self.vr_sys.getEyeTrackingData()
        # fix origin and direction 
        if not self.eye_tracking_data[0]: # combined data invalid
            self.eye_tracking_data[3] = [-1, -1, -1]
            self.eye_tracking_data[4] = [-1, -1, -1]
        if not self.eye_tracking_data[1]: # left data invalid
            self.eye_tracking_data[5] = [-1, -1, -1]
            self.eye_tracking_data[6] = [-1, -1, -1]
        if not self.eye_tracking_data[2]: # right data invalid
            self.eye_tracking_data[7] = [-1, -1, -1]
            self.eye_tracking_data[8] = [-1, -1, -1]

    def poll_vr_events(self):
        """
        Returns VR event data as list of lists.
        List is empty if all events are invalid. Components of a single event:
        controller: 0 (left_controller), 1 (right_controller)
        button_idx: any valid idx in EVRButtonId enum in openvr.h header file
        press: 0 (unpress), 1 (press)
        """

        self.vr_event_data = self.vr_sys.pollVREvents()
        # Enforce store_first_button_press_per_frame option, if user has enabled it
        if self.vr_settings.store_only_first_button_event:
            temp_event_data = []
            # Make sure we only store the first (button, press) combo of each type
            event_set = set()
            for ev_data in self.vr_event_data:
                controller, button_idx, _ = ev_data
                key = (controller, button_idx)
                if key not in event_set:
                    temp_event_data.append(ev_data)
                    event_set.add(key)
            self.vr_event_data = temp_event_data[:]

            if len(self.vr_event_data) != 0:
                for ev_data in self.vr_event_data:
                    controller, button_idx, pressed = ev_data

        return self.vr_event_data

    def get_vr_events(self):
        """
        Returns the VR events processed by the simulator
        """
        return self.vr_event_data

    def query_vr_event(self, controller: str, action: str) -> bool:
        """
        Queries system for a VR event, and returns true if that event happened this frame
        Args:
            controller (str): device to query for - either left_controller or right_controller
            action (str): an action name listed in "action_button_map" dictionary for the current device in the vr_config.yml
        
        """
        # Return false if any of input parameters are invalid
        if (
            controller not in ["left_controller", "right_controller"]
            or action not in self.vr_settings.action_button_map.keys()
        ):
            return False

        # Search through event list to try to find desired event
        controller_id = 0 if controller == "left_controller" else 1
        button_idx, press_id = self.vr_settings.action_button_map[action]
        for ev_data in self.vr_event_data:
            if controller_id == ev_data[0] and button_idx == ev_data[1] and press_id == ev_data[2]:
                return True
        # Return false if event was not found this frame
        return False

    def get_action_button_state(self, controller: str, action: str, vr_data: VrData) -> bool:
        """This function can be used to extract the _state_ of a button from the vr_data's buttons_pressed vector.
        If only key press/release events are required, use the event polling mechanism. This function is meant for
        providing access to the continuous pressed/released state of the button.

        Args:
            controller (str): one of left_controller or right_controller
            action (str): action string (see vr_config.yaml)
            vr_data: VrData class that holds device data
        Return:
            bool: whether the action has happened on the controller
        """
        # Find the controller and find the button mapping for this action in the config.
        if (
                controller not in ["left_controller", "right_controller"]
                or action not in self.vr_settings.action_button_map.keys()
        ):
            return False

        # Find the button index for this action from the config.
        button_idx, _ = self.vr_settings.action_button_map[action]

        # Get the bitvector corresponding to the buttons currently pressed on the controller.
        controller_button_data = vr_data.query(f"{controller}_button")
        buttons_pressed = int(controller_button_data[0] * controller_button_data[4])

        # Extract and return the value of the bit corresponding to the button.
        return bool(buttons_pressed & (1 << button_idx))

    def set_vr_start_position(self, start_position: Optional[List[float]]=None, vr_start_height_offset: Optional[float]=None):
        """
        Sets the starting position of the VR system in iGibson space

        Args:
            start_position (List[float]): position to start VR system at. Default is None
            vr_start_height_offset (float): starting height offset. If None, uses absolute height from start_position
        """

        # The VR headset will actually be set to this position during the first frame.
        # This is because we need to know where the headset is in space when it is first picked
        # up to set the initial offset correctly.
        self.vr_start_position = start_position
        # This value can be set to specify a height offset instead of an absolute height.
        # We might want to adjust the height of the camera based on the height of the person using VR,
        # but still offset this height. When this option is not None it offsets the height by the amount
        # specified instead of overwriting the VR system height output.
        self.vr_start_height_offset = vr_start_height_offset

    def initialize_vr_render(self):
        img = np.zeros((4, 4, 4))
        self.vr_text_left_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.vr_text_left_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)

        self.vr_text_right_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.vr_text_right_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)
