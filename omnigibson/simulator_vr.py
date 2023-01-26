import time
import numpy as np

from time import sleep
from typing import List
from OpenGL.GL import *
from omnigibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omnigibson.utils.vr_utils import VrSettings, VrHUDOverlay, VrStaticImageOverlay
from omnigibson.render.mesh_renderer import VRRendererContext
from omnigibson.robots.BR33 import HAND_BASE_ROTS
from omnigibson.robots.manipulation_robot import IsGraspingState
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
            viewer_width=1280,
            viewer_height=720,
            vertical_fov=90,
            device_idx=None,
            apply_transitions=False
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
        self.vr_settings = VrSettings(use_vr=True)
        self.vr_overlay_initialized = False
        self.vr_start_pos = None
        self.max_haptic_duration = 4000
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
            vertical_fov,
            device_idx,
            apply_transitions
        )

        # Get expected number of vsync frames per iGibson frame Note: currently assumes a 90Hz VR system
        self.vsync_frame_num = int(round(rendering_dt / self.vsync_frame_dur))

        # Total amount of time we want non-blocking actions to take each frame
        # Leave a small amount of time before the last vsync, just in case we overrun
        self.non_block_frame_time = (self.vsync_frame_num - 1) * self.vsync_frame_dur + (
            5e-3 if self.vr_settings.curr_device == "OCULUS" else 10e-3
        )

        self.rendering_settings = MeshRendererSettings()
        self.rendering_dt = rendering_dt
        self.main_vr_robot = None
        self.vr_hud = None
        self.vr_text_left_id, self.vr_text_right_id = None, None
        self.vr_sys = None

    def initialize_vr(self):
        self.vr_sys = VRRendererContext.VRRendererContext(
            self.viewer_width,
            self.viewer_height,
            int(self.rendering_settings.glfw_gl_version[0]),
            int(self.rendering_settings.glfw_gl_version[1]),
            self.rendering_settings.show_glfw_window,
            self.rendering_settings.fullscreen,
        )
        self.vr_sys.init()
        self.vr_sys.initVR(self.vr_sys.hasEyeTrackingSupport())
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
        # First sync VR compositor - this is where Oculus blocks (as opposed to Vive, which blocks in update_vr_data)
        self.sync_vr_compositor()
        # Note: this should only be called once per frame - use get_vr_events to read the event data list in
        # subsequent read operations
        self.poll_vr_events()
        # This is necessary to fix the eye tracking value for the current frame, since it is multi-threaded
        self.fix_eye_tracking_value()
        # Move user to their starting location
        self.perform_vr_start_pos_move()
        # Update VR data and wait until 3ms before the next vsync
        self.vr_sys.updateVRData()
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
        ps, nps = super().step()
        # update collision status of BR hand
        if self.main_vr_robot:
            self.main_vr_robot.update_hand_contact_info()
        physics_dur = time.perf_counter() - omni_step_start_time
        # render to headset
        render_start_time = time.perf_counter()
        self.render_vr()
        render_dur = time.perf_counter() - render_start_time
        # Sleep until last possible Vsync
        pre_sleep_dur = outside_step_dur + physics_dur + render_dur
        sleep_start_time = time.perf_counter()
        if pre_sleep_dur < self.non_block_frame_time:
            sleep(self.non_block_frame_time - pre_sleep_dur)
        sleep_dur = time.perf_counter() - sleep_start_time
        vr_system_dur = self.step_vr_system()

        # Calculate final frame duration
        # Make sure it is non-zero for FPS calculation (set to max of 1000 if so)
        frame_dur = max(1e-3, pre_sleep_dur + sleep_dur + vr_system_dur)

        # Set variables for data saving and replay
        self.last_physics_step = physics_dur
        self.last_render_timestep = render_dur
        self.last_frame_dur = frame_dur

        if print_stats:
            print("Frame number {} statistics (ms)".format(self.frame_count))
            print("Total out-of-step duration: {}".format(outside_step_dur * 1000))
            print("Total physics step duration: {}".format(ps * 1000))
            print("Total non physics step duration: {}".format(nps * 1000))
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
        self.vr_sys.postRenderVRForEye("left", self.vr_text_left_id)
        self.vr_sys.postRenderVRForEye("right", self.vr_text_right_id)

    def vr_system_update(self):
        """
        Updates the VR system for a single frame. This includes moving the vr offset,
        adjusting the user's height based on button input, and triggering haptics.
        """
        # Update VR offset using appropriate controller
        if self.vr_settings.touchpad_movement:
            vr_offset_device = "{}_controller".format(self.vr_settings.movement_controller)
            is_valid, _, _ = self.get_data_for_vr_device(vr_offset_device)
            if is_valid:
                _, touch_x, touch_y, _ = self.get_button_data_for_controller(vr_offset_device)
                new_offset = calc_offset(
                    self, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device
                )
                self.set_vr_offset(new_offset)

        # Adjust user height based on y-axis (vertical direction) touchpad input
        vr_height_device = "left_controller" if self.vr_settings.movement_controller == "right" else "right_controller"
        is_height_valid, _, _ = self.get_data_for_vr_device(vr_height_device)
        if is_height_valid:
            curr_offset = self.get_vr_offset()
            hmd_height = self.get_hmd_world_pos()[2]
            _, _, height_y, _ = self.get_button_data_for_controller(vr_height_device)
            if height_y < -0.7:
                vr_z_offset = -0.01
                if hmd_height + curr_offset[2] + vr_z_offset >= self.vr_settings.height_bounds[0]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
            elif height_y > 0.7:
                vr_z_offset = 0.01
                if hmd_height + curr_offset[2] + vr_z_offset <= self.vr_settings.height_bounds[1]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
        # Update haptics for body and hands
        if self.main_vr_robot:
            # Check for body haptics
            if self.main_vr_robot.part_is_in_contact["body"]:
                for controller in ["left_controller", "right_controller"]:
                    is_valid, _, _ = self.get_data_for_vr_device(controller)
                    if is_valid:
                        # Use 90% strength for body to warn user of collision with wall
                        self.trigger_haptic_pulse(controller, 0.9)

            # Check for hand haptics
            for hand_device, hand_name in [("left_controller", "lh"), ("right_controller", "rh")]:
                is_valid, _, _ = self.get_data_for_vr_device(hand_device)
                if is_valid:
                    if (
                        self.main_vr_robot.part_is_in_contact[hand_name]
                        or self.main_vr_robot.is_grasping(hand_name) == IsGraspingState.TRUE
                    ):
                        # Only use 30% strength for normal collisions, to help add realism to the experience
                        self.trigger_haptic_pulse(hand_device, 0.3)

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
            is_valid, trans, rot = self.get_data_for_vr_device(device)
            device_data = [is_valid, trans.tolist(), rot.tolist()]
            device_data.extend(self.get_device_coordinate_system(device))
            v[device] = device_data
            if device in VR_CONTROLLERS:
                v["{}_button".format(device)] = self.get_button_data_for_controller(device)

        # Store final rotations of hands, with model rotation applied
        for hand in ["right", "left"]:
            # Base rotation quaternion
            base_rot = HAND_BASE_ROTS[hand]
            # Raw rotation of controller
            controller_rot = v["{}_controller".format(hand)][2]
            # Use dummy translation to calculation final rotation
            final_rot = T.pose_transform([0, 0, 0], controller_rot, [0, 0, 0], base_rot)[1]
            v["{}_controller".format(hand)].append(final_rot)

        is_valid, torso_trans, torso_rot = self.get_data_for_vr_tracker(self.vr_settings.torso_tracker_serial)
        v["torso_tracker"] = [is_valid, torso_trans, torso_rot]
        v["eye_data"] = self.get_eye_tracking_data()
        v["event_data"] = self.get_vr_events()
        reset_actions = []
        for controller in VR_CONTROLLERS:
            reset_actions.append(self.query_vr_event(controller, "reset_agent"))
        v["reset_actions"] = reset_actions
        v["vr_positions"] = [self.get_vr_pos().tolist(), list(self.get_vr_offset())]
        return VrData(v)

    def sync_vr_compositor(self):
        """
        Sync VR compositor.
        """
        self.vr_sys.postRenderVR(True)

    def perform_vr_start_pos_move(self):
        """
        Sets the VR position on the first step iteration where the hmd tracking is valid. Not to be confused
        with self.set_vr_start_pos, which simply records the desired start position before the simulator starts running.
        """
        # Update VR start position if it is not None and the hmd is valid
        # This will keep checking until we can successfully set the start position
        if self.vr_start_pos:
            hmd_is_valid, _, _, _ = self.vr_sys.getDataForVRDevice("hmd")
            if hmd_is_valid:
                offset_to_start = np.array(self.vr_start_pos) - self.get_hmd_world_pos()
                if self.vr_height_offset is not None:
                    offset_to_start[2] = self.vr_height_offset
                self.set_vr_offset(offset_to_start)
                self.vr_start_pos = None

    def fix_eye_tracking_value(self):
        """
        Calculates and fixes eye tracking data to its value during step(). This is necessary, since multiple
        calls to get eye tracking data return different results, due to the SRAnipal multithreaded loop that
        runs in parallel to the iGibson main thread
        """
        self.eye_tracking_data = self.vr_sys.getEyeTrackingData()

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
        if self.vr_settings.store_only_first_event_per_button:
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
                for event in self.vr_event_data:
                    controller, button_idx, pressed = ev_data
                    pressed_str = "pressed" if pressed else "unpressed"
                    print("Controller %d button %d %s" % (controller, button_idx, pressed_str))

        return self.vr_event_data

    def get_vr_events(self):
        """
        Returns the VR events processed by the simulator
        """
        return self.vr_event_data

    def query_vr_event(self, controller, action):
        """
        Queries system for a VR event, and returns true if that event happened this frame
        :param controller: device to query for - can be left_controller or right_controller
        :param action: an action name listed in "action_button_map" dictionary for the current device in the vr_config.yml
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

    def get_data_for_vr_device(self, device_name: str) -> List:
        """
        Call this after step - returns all VR device data for a specific device
        Returns is_valid (indicating validity of data), translation and rotation in Gibson world space
        Args:
             device_name (str): one of hmd, left_controller or right_controller
        Returns:
            List[bool, List[float], List[float]]: isvalid, translation and rotation
        """

        # Use fourth variable in list to get actual hmd position in space
        is_valid, translation, rotation, _ = self.vr_sys.getDataForVRDevice(device_name)
        if not is_valid:
            translation = np.array([0, 0, 0])
            rotation = np.array([0, 0, 0, 1])
        return [is_valid, translation, rotation]

    def get_data_for_vr_tracker(self, tracker_serial_number: str) -> List:
        """
        Returns the data for a tracker with a specific serial number. See vr_config.yaml for how to find serial number
        Args:
             tracker_serial_number (str): the serial number of the tracker
        Returns:
            List[bool, List[float], List[float]]: isvalid, translation and rotation
        """

        if not tracker_serial_number:
            return [False, np.zeros(3), np.zeros(4)]

        tracker_data = self.vr_sys.getDataForVRTracker(tracker_serial_number)
        # Set is_valid to false, and assume the user will check for invalid data
        if not tracker_data:
            return [False, np.array([0, 0, 0]), np.array([0, 0, 0, 1])]

        is_valid, translation, rotation = tracker_data
        return [is_valid, translation, rotation]

    def get_hmd_world_pos(self) -> List[float]:
        """
        Get world position of HMD without offset
        """

        return self.vr_sys.getDataForVRDevice("hmd")[3]

    def get_button_data_for_controller(self, controller_name: str) -> List:
        """
        Call this after getDataForVRDevice - returns analog data for a specific controller
        Returns trigger_fraction, touchpad finger position x, touchpad finger position y
        Data is only valid if isValid is true from previous call to getDataForVRDevice
        Trigger data: 1 (closed) <------> 0 (open)
        Analog data: X: -1 (left) <-----> 1 (right) and Y: -1 (bottom) <------> 1 (top)

        Args:
             controller_name (str): one of left_controller or right_controller
        """

        # Test for validity when acquiring button data
        if self.get_data_for_vr_device(controller_name)[0]:
            trigger_fraction, touch_x, touch_y, buttons_pressed = self.vr_sys.getButtonDataForController(
                controller_name
            )
        else:
            trigger_fraction, touch_x, touch_y, buttons_pressed = 0.0, 0.0, 0.0, 0
        return [trigger_fraction, touch_x, touch_y, buttons_pressed]

    def get_action_button_state(self, controller: str, action, vr_data: VrData) -> bool:
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
        buttons_pressed = int(vr_data.query("%s_button" % controller)[3])

        # Extract and return the value of the bit corresponding to the button.
        return bool(buttons_pressed & (1 << button_idx))

    def get_scroll_input(self) -> int:
        """
        Gets scroll input. This uses the non-movement-controller, and determines whether
        the user wants to scroll by testing if they have pressed the touchpad, while keeping
        their finger on the left/right of the pad.

        Return:
            int: 1 for up, 0 for down, -1 for no scroll
        """
        mov_controller = self.vr_settings.movement_controller
        other_controller = "right" if mov_controller == "left" else "left"
        other_controller = "{}_controller".format(other_controller)
        # Data indicating whether user has pressed top or bottom of the touchpad
        _, touch_x, _ = self.vr_sys.getButtonDataForController(other_controller)
        # Detect no touch in extreme regions of x axis
        if 0.7 < touch_x <= 1.0:
            return 1
        elif -0.7 > touch_x >= -1.0:
            return 0
        else:
            return -1

    def get_eye_tracking_data(self):
        """
        Returns eye tracking data as list of lists. Order: is_valid, gaze origin, gaze direction, gaze point,
        left pupil diameter, right pupil diameter (both in millimeters)
        Call after getDataForVRDevice, to guarantee that latest HMD transform has been acquired
        """
        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.eye_tracking_data
        # Set other values to 0 to avoid very small/large floating point numbers
        if not is_valid:
            return [False, [0, 0, 0], [0, 0, 0], 0, 0, [0, 0], [0, 0]]
        else:
            return [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter]

    def set_vr_start_pos(self, start_pos=None, vr_height_offset=None):
        """
        Sets the starting position of the VR system in iGibson space

        Args:
            start_pos: position to start VR system at. Default is None
            vr_height_offset: starting height offset. If None, uses absolute height from start_pos
        """

        # The VR headset will actually be set to this position during the first frame.
        # This is because we need to know where the headset is in space when it is first picked
        # up to set the initial offset correctly.
        self.vr_start_pos = start_pos
        # This value can be set to specify a height offset instead of an absolute height.
        # We might want to adjust the height of the camera based on the height of the person using VR,
        # but still offset this height. When this option is not None it offsets the height by the amount
        # specified instead of overwriting the VR system height output.
        self.vr_height_offset = vr_height_offset

    def set_vr_pos(self, pos: List[float], keep_height: bool = False):
        """
        Sets the world position of the VR system in iGibson space

        Args:
            pos (List[float]): position to set VR system to
            keep_height (bool): whether the current VR height should be kept. Default is False.
        """

        offset_to_pos = np.array(pos) - self.get_hmd_world_pos()
        if keep_height:
            curr_offset_z = self.get_vr_offset()[2]
            self.set_vr_offset([offset_to_pos[0], offset_to_pos[1], curr_offset_z])
        else:
            self.set_vr_offset(offset_to_pos)

    def get_vr_pos(self) -> List[float]:
        """
        Gets the world position of the VR system in iGibson space.
        """
        return self.get_hmd_world_pos() + np.array(self.get_vr_offset())

    def set_vr_offset(self, pos: List[float]):
        """
        Sets the translational offset of the VR system (HMD, left & right controller) from world space coordinates.
        Can be used for many things, including adjusting height and teleportation-based movement
        Args:
             pos (List[float]): a list of three floats corresponding to x, y, z in Gibson coordinate space
        """

        self.vr_sys.setVROffset(-pos[1], pos[2], -pos[0])

    def get_vr_offset(self) -> List[float]:
        """
        Gets the current VR offset vector in list form: x, y, z (in iGibson coordinates)
        """

        x, y, z = self.vr_sys.getVROffset()
        return [x, y, z]

    def get_device_coordinate_system(self, device: str):
        """
        Gets the direction vectors representing the device's coordinate system in list form: x, y, z (in Gibson coordinates)
        List contains "right", "up" and "forward" vectors in that order
        Args:
            device: one of "hmd", "left_controller" or "right_controller"
        """

        vec_list = []

        coordinate_sys = self.vr_sys.getDeviceCoordinateSystem(device)
        for dir_vec in coordinate_sys:
            vec_list.append(dir_vec)

        return vec_list

    def trigger_haptic_pulse(self, device: str, strength: float):
        """
        Triggers a haptic pulse of the specified strength (0 is weakest, 1 is strongest)
        Args:
            device (str): device to trigger haptic for - can be any one of [left_controller, right_controller]
            strength (float): strength of haptic pulse (0 is weakest, 1 is strongest)
        """
        assert device in ["left_controller", "right_controller"]
        self.vr_sys.triggerHapticPulseForDevice(device, int(self.max_haptic_duration * strength))

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
