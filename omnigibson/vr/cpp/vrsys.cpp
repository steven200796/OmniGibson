#include "vrsys.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>


# pragma region Public Methods

/// @brief initialize GLFW and OpenVR
/// @param useEyeTracking whether to use eye tracking (only supports windows)
void VRSys::init(bool useEyeTracking=false) {
	// Initialize GLFW
	if (!glfwInit()) {
        fprintf(stderr, "[VRSys] Failed to initialize GLFW!\n");
        exit(EXIT_FAILURE);
    }
    glfwDefaultWindowHints();
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	this->window = glfwCreateWindow(1, 1, "OG GLFW", NULL, NULL);
    if (this->window == NULL) {
        fprintf(stderr, "[VRSys] Failed to create GLFW window!\n");
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(this->window);
    glfwSwapInterval(0);

    // Load all OpenGL function pointers through GLAD
    if (!gladLoadGL()) {
        fprintf(stderr, "[VRSys] Failed to load OpenGL function pointers through GLAD!\n");
        exit(EXIT_FAILURE);
    }	
	
	// Initialize OpenVR
	if (!vr::VR_IsRuntimeInstalled()) {
		fprintf(stderr, "[VRSys] OpenVR runtime not installed!\n");
		exit(EXIT_FAILURE);
	}

	vr::EVRInitError eError = vr::VRInitError_None;
	m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None) {
		fprintf(stderr, "[VRSys] Failed to initialize VR runtime: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		exit(EXIT_FAILURE);
	}
	if (!vr::VRCompositor()) {
		fprintf(stderr, "[VRSys] Failed to intialize VR compositor!\n");
	}

	leftEyeProj = getHMDEyeProjection(vr::Eye_Left);
	leftEyePose = getHMDEyePose(vr::Eye_Left);
	rightEyeProj = getHMDEyeProjection(vr::Eye_Right);
	rightEyePose = getHMDEyePose(vr::Eye_Right);

	// Set gibToVr and vrToGib matrices
	gibToVr[0] = glm::vec4(0.0, 0.0, -1.0, 0.0);
	gibToVr[1] = glm::vec4(-1.0, 0.0, 0.0, 0.0);
	gibToVr[2] = glm::vec4(0.0, 1.0, 0.0, 0.0);
	gibToVr[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);

	vrToGib[0] = glm::vec4(0.0, -1.0, 0.0, 0.0);
	vrToGib[1] = glm::vec4(0.0, 0.0, 1.0, 0.0);
	vrToGib[2] = glm::vec4(-1.0, 0.0, 0.0, 0.0);
	vrToGib[3] = glm::vec4(0.0, 0.0, 0.0, 1.0);

	vrOffsetVec = glm::vec3(0, 0, 0);	// No VR system offset by default

	// Only activate eye tracking on Windows
	#ifdef WIN32
		this->useEyeTracking = useEyeTracking;
		if (useEyeTracking) {
			shouldShutDownEyeTracking = false;
			if (!ViveSR::anipal::Eye::IsViveProEye()) {
				fprintf(stderr, "[VRSys] This HMD does not support eye-tracking!\n");
				exit(EXIT_FAILURE);
			}
			int anipalError = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
			switch (anipalError) {
				case ViveSR::Error::WORK:
					break;
				case ViveSR::Error::RUNTIME_NOT_FOUND:
					fprintf(stderr, "[VRSys] SRAnipal runtime not installed!\n");
					exit(EXIT_FAILURE);
				default:
					fprintf(stderr, "[VRSys] Failed to initialize SRAnipal!\n");
					exit(EXIT_FAILURE);
			}
			// Launch a thread to poll data from the SRAnipal SDK
			// We poll data asynchronously so as to not slow down the VR rendering loop
			eyeTrackingThread = new std::thread(&VRSys::pollAnipal, this);
		}
	#endif
}



/// @brief Releases and cleans up VR and gl system
void VRSys::release() {
	vr::VR_Shutdown();
	m_pHMD = NULL;
	glfwTerminate();

	#ifdef WIN32
		if (this->useEyeTracking) {
			this->shouldShutDownEyeTracking = true;
			eyeTrackingThread->join();
		}
	#endif
}



/// @brief Returns the projection and view matrices for the left and right eyes
/// @return Left P, left V, right P, right V
/// @note Call before rendering so the camera is set properly
py::list VRSys::preRender() {
	py::array_t<float> leftEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(leftEyeProj));
	py::array_t<float> rightEyeProjNp = py::array_t<float>({ 4,4 }, glm::value_ptr(rightEyeProj));

	glm::mat4 worldToHead = glm::inverse(hmdData.deviceTransform);

	leftEyeView = leftEyePose * worldToHead;
	leftEyeView = leftEyeView * gibToVr;
	// transpose the matrix before returning
	py::array_t<float> leftEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(leftEyeView)));

	rightEyeView = rightEyePose * worldToHead;
	rightEyeView = rightEyeView * gibToVr;
	py::array_t<float> rightEyeViewNp = py::array_t<float>({ 4,4 }, glm::value_ptr(glm::transpose(rightEyeView)));

	py::list eyeMats;
	eyeMats.append(leftEyeProjNp);
	eyeMats.append(leftEyeViewNp);
	eyeMats.append(rightEyeProjNp);
	eyeMats.append(rightEyeViewNp);

	return eyeMats;
}



/// @brief updated scene texture to display
/// @param lefttexID OpenGL texture ID for left eye
/// @param rigTtTexID OpenGL texture ID for right eye
void VRSys::render(GLuint leftTexID, GLuint rightTexID) {
	vr::Texture_t leftEyeTex = { (void*)(uintptr_t)leftTexID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
	vr::Texture_t rightEyeTex = { (void*)(uintptr_t)rightTexID, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
	vr::EVRCompositorError err = vr::VRCompositorError_None;
	err = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTex);
	// 0 is no error, 101 is no focus (happens at start of rendering)
	if (err != 0 && err != 101) {
		fprintf(stderr, "[VRSys] Compositor error: %d\n", err);
	}
	err = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTex);
	if (err != 0 && err != 101) {
		fprintf(stderr, "[VRSys] Compositor error: %d\n", err);
	}
	vr::VRCompositor()->PostPresentHandoff();	// Tell the compositor to begin work immediately
	glFlush();	// Flush rendering queue to get GPU working ASAP
}



/// @brief Polls for VR events, such as button presses, guaranteed to only return valid events
/// TIMELINE: Call before simulator.step()
/// @return list of action data
py::list VRSys::pollVREvents() {
	vr::VREvent_t vrEvent;
	py::list eventData;

	while (m_pHMD->PollNextEvent(&vrEvent, sizeof(vrEvent))) {
		int controller_type = -1, action_type = -1;
		vr::ETrackedDeviceClass trackedDeviceClass = m_pHMD->GetTrackedDeviceClass(vrEvent.trackedDeviceIndex);
		// Exit if we found a non-controller event
		if (trackedDeviceClass != vr::ETrackedDeviceClass::TrackedDeviceClass_Controller)
			continue;

		int button_idx = vrEvent.data.controller.button;

		vr::ETrackedControllerRole role = m_pHMD->GetControllerRoleForTrackedDeviceIndex(vrEvent.trackedDeviceIndex);
		if (role == vr::TrackedControllerRole_Invalid) 			controller_type = -1;
		else if (role == vr::TrackedControllerRole_LeftHand) 	controller_type = 0;
		else if (role == vr::TrackedControllerRole_RightHand) 	controller_type = 1;

		// Both ButtonPress and ButtonTouch count as "press" (same goes for unpress/untouch)
		int event_type = vrEvent.eventType;
		if (event_type == vr::VREvent_ButtonUnpress || event_type == vr::VREvent_ButtonUntouch)	{
			action_type = 0;
		}
		else if (event_type == vr::VREvent_ButtonPress || event_type == vr::VREvent_ButtonTouch){
			action_type = 1;
		}	
		// Only record data if everything is valid
		if (!(controller_type == -1 || button_idx == -1 || action_type == -1)) {
			py::list singleEventData;
			singleEventData.append(controller_type);
			singleEventData.append(button_idx);
			singleEventData.append(action_type);
			eventData.append(singleEventData);
		}
	}

	return eventData;
}



/// @brief Calls WaitGetPoses and GetControllerState to update all device poses and button states
void VRSys::pollVRPosesAndStates() {
	hmdData.isValidData = false;
	leftControllerData.isValidData = false;
	rightControllerData.isValidData = false;

	vr::VRControllerState_t controllerState;

	vr::TrackedDevicePose_t trackedDevicesPose[vr::k_unMaxTrackedDeviceCount];
	vr::VRCompositor()->WaitGetPoses(trackedDevicesPose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	for (unsigned int idx = 0; idx < vr::k_unMaxTrackedDeviceCount; idx++) {
		if (!trackedDevicesPose[idx].bPoseIsValid || !m_pHMD->IsTrackedDeviceConnected(idx)) continue;

		vr::HmdMatrix34_t poseMat = trackedDevicesPose[idx].mDeviceToAbsoluteTracking;
		glm::vec3 devicePositionWithoutOffset = getPositionFromSteamVRMatrix(poseMat);
		glm::vec3 devicePositionWithOffset = devicePositionWithoutOffset + vrOffsetVec;
		setSteamVRMatrixPosition(devicePositionWithOffset, poseMat);

		vr::ETrackedDeviceClass trackedDeviceClass = m_pHMD->GetTrackedDeviceClass(idx);
		if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_HMD) {
			hmdData.index = idx;
			hmdData.isValidData = true;
			hmdData.deviceTransform = convertSteamVRMatrixToGlmMat4(poseMat);
			hmdData.devicePos = getPositionFromSteamVRMatrix(poseMat);
			hmdData.deviceRot = getRotationFromSteamVRMatrix(poseMat);
			hmdActualPosition = devicePositionWithoutOffset;
		}
		else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_Controller) {
			vr::ETrackedControllerRole role = m_pHMD->GetControllerRoleForTrackedDeviceIndex(idx);
			if (role == vr::TrackedControllerRole_Invalid) {
				continue;
			}

			int trigger_index, touchpad_index;
			// Figure out indices that correspond with trigger and trackpad axes. Index used to read into VRControllerState_t struct array of axes.
			for (int i = 0; i < vr::k_unControllerStateAxisCount; i++) {
				int axisType = m_pHMD->GetInt32TrackedDeviceProperty(idx, (vr::ETrackedDeviceProperty)(vr::Prop_Axis0Type_Int32 + i));
				if (axisType == vr::EVRControllerAxisType::k_eControllerAxis_Trigger) {
					trigger_index = i;
				}
				// Detect trackpad on HTC Vive controller and Joystick on Oculus controller
				else if (axisType == vr::EVRControllerAxisType::k_eControllerAxis_TrackPad || axisType == vr::EVRControllerAxisType::k_eControllerAxis_Joystick) {
					touchpad_index = i;
				}
			}

			bool getControllerDataResult = m_pHMD->GetControllerState(idx, &controllerState, sizeof(controllerState));
			if (role == vr::TrackedControllerRole_LeftHand) {
				leftControllerData.index = idx;
				leftControllerData.isValidData = getControllerDataResult;
				leftControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(poseMat);
				leftControllerData.devicePos = getPositionFromSteamVRMatrix(poseMat);
				leftControllerData.deviceRot = getRotationFromSteamVRMatrix(poseMat);

				leftControllerData.triggerFraction = controllerState.rAxis[trigger_index].x;
				leftControllerData.touchpadAnalogVector = glm::vec2(controllerState.rAxis[touchpad_index].x, controllerState.rAxis[touchpad_index].y);
				leftControllerData.buttonsPressed = controllerState.ulButtonPressed;
			}
			else if (role == vr::TrackedControllerRole_RightHand) {
				rightControllerData.index = idx;
				rightControllerData.isValidData = getControllerDataResult;
				rightControllerData.deviceTransform = convertSteamVRMatrixToGlmMat4(poseMat);
				rightControllerData.devicePos = getPositionFromSteamVRMatrix(poseMat);
				rightControllerData.deviceRot = getRotationFromSteamVRMatrix(poseMat);

				rightControllerData.triggerFraction = controllerState.rAxis[trigger_index].x;
				rightControllerData.touchpadAnalogVector = glm::vec2(controllerState.rAxis[touchpad_index].x, controllerState.rAxis[touchpad_index].y);
				rightControllerData.buttonsPressed = controllerState.ulButtonPressed;
			}
		}
		else if (trackedDeviceClass == vr::ETrackedDeviceClass::TrackedDeviceClass_GenericTracker) {
			// Identify generic trackers by their serial number
			char serial_name[vr::k_unMaxPropertyStringSize];
			uint32_t serial_name_len = m_pHMD->GetStringTrackedDeviceProperty(idx, vr::ETrackedDeviceProperty::Prop_SerialNumber_String, serial_name, vr::k_unMaxPropertyStringSize);
			std::string serial(serial_name, serial_name_len-1);

			if (this->trackerNamesToData.find(serial) != this->trackerNamesToData.end()) {
				this->trackerNamesToData[serial].index = idx;
				this->trackerNamesToData[serial].isValidData = true;
				this->trackerNamesToData[serial].deviceTransform = convertSteamVRMatrixToGlmMat4(poseMat);
				this->trackerNamesToData[serial].devicePos = getPositionFromSteamVRMatrix(poseMat);
				this->trackerNamesToData[serial].deviceRot = getRotationFromSteamVRMatrix(poseMat);
			}
			else {
				DeviceData trackerData;
				trackerData.index = idx;
				trackerData.isValidData = true;
				trackerData.deviceTransform = convertSteamVRMatrixToGlmMat4(poseMat);
				trackerData.devicePos = getPositionFromSteamVRMatrix(poseMat);
				trackerData.deviceRot = getRotationFromSteamVRMatrix(poseMat);
				this->trackerNamesToData[serial] = trackerData;
			}
		}
	}
}



/// @brief Get button data for a specific controller
/// @param controllerType: either left_controller or right_controller
/// @return list containing trigger fraction, touchpad position x, touchpad position y, pressed buttons bitvector
py::list VRSys::getControllerButtonData(char* controllerType) {
    DeviceData device_data;
	if (!strcmp(controllerType, "left_controller")) 
		device_data = leftControllerData;
	else if (!strcmp(controllerType, "right_controller")) 
		device_data = rightControllerData;

	py::list buttonData;
	buttonData.append(device_data.isValidData);
	if (device_data.isValidData) {
		buttonData.append(device_data.triggerFraction);
		buttonData.append(device_data.touchpadAnalogVector.x);
		buttonData.append(device_data.touchpadAnalogVector.y);
		buttonData.append(device_data.buttonsPressed);
	}
	else {	// return 0 for invalid data
		buttonData.append(0);
		buttonData.append(0);
		buttonData.append(0);
		buttonData.append(0);
	}
	return buttonData;
}



/// @brief get device poses information
/// @param deviceID either hmd, left_controller, right_controller, or tracker serial number
/// @return list in order: isValidData, position, rotation, hmdActualPosition (valid only if hmd)
py::list VRSys::getDevicePose(char* deviceID) {
	bool isValid = false;

	py::array_t<float> positionData;
	py::array_t<float> rotationData;
	py::array_t<float> hmdActualPositionData;

	if (!strcmp(deviceID, "hmd")) {
		glm::vec3 transformedPos(vrToGib * glm::vec4(hmdData.devicePos, 1.0));
		positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
		rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * hmdData.deviceRot));
		glm::vec3 transformedHmdPos(vrToGib * glm::vec4(hmdActualPosition, 1.0));
		hmdActualPositionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedHmdPos));
		isValid = hmdData.isValidData;
	}
	else if (!strcmp(deviceID, "left_controller")) {
		glm::vec3 transformedPos(vrToGib * glm::vec4(leftControllerData.devicePos, 1.0));
		positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
		rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * leftControllerData.deviceRot));
		isValid = leftControllerData.isValidData;
	}
	else if (!strcmp(deviceID, "right_controller")) {
		glm::vec3 transformedPos(vrToGib * glm::vec4(rightControllerData.devicePos, 1.0));
		positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
		rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * rightControllerData.deviceRot));
		isValid = rightControllerData.isValidData;
	}
	else {	// generic tracker
		std::string trackerSerialNumber = std::string(deviceID);
		// Return empty tracker data list if the tracker serial number is invalid
		if (this->trackerNamesToData.find(trackerSerialNumber) != this->trackerNamesToData.end()) {
			DeviceData currTrackerData = this->trackerNamesToData[trackerSerialNumber];
			glm::vec3 transformedPos(vrToGib * glm::vec4(currTrackerData.devicePos, 1.0));
			positionData = py::array_t<float>({ 3, }, glm::value_ptr(transformedPos));
			rotationData = py::array_t<float>({ 4, }, glm::value_ptr(vrToGib * currTrackerData.deviceRot));
			isValid = currTrackerData.isValidData;
		}
	}

	py::list deviceData;
	deviceData.append(isValid);
	deviceData.append(positionData);
	deviceData.append(rotationData);
	deviceData.append(hmdActualPositionData);

	return deviceData;
}



/// @brief Gets normalized vectors representing the device coordinate system
/// @param device hmd, left_controller or right_controller
/// @return [right, up, forward] relative to the device in OG coordinate system
py::list VRSys::getDeviceCoordinateSystem(char* device) {
	py::list vecList;
	glm::mat4 deviceTransform;

	if (!strcmp(device, "hmd")) 
		deviceTransform = hmdData.deviceTransform;
	else if (!strcmp(device, "left_controller")) 
		deviceTransform = leftControllerData.deviceTransform;
	else if (!strcmp(device, "right_controller"))
		deviceTransform = rightControllerData.deviceTransform;

	for (int i = 0; i < 3; i++) {
		glm::vec3 transformedVrDir = getVec3ColFromMat4(i, deviceTransform);
		if (i == 2) {
			transformedVrDir = transformedVrDir * -1.0f;
		}
		glm::vec3 transformedGibDir = glm::normalize(glm::vec3(vrToGib * glm::vec4(transformedVrDir, 1.0)));

		py::list vec;
		vec.append(transformedGibDir.x);
		vec.append(transformedGibDir.y);
		vec.append(transformedGibDir.z);
		vecList.append(vec);
	}

	return vecList;
}



/// @brief Queries eye tracking data and returns to user
/// @return list containing eye tracking data
/// @note Call after getDevicePose, since this relies on knowing latest HMD transform
py::list VRSys::getEyeTrackingData() {
	py::list eyeData, combinedOrigin, combinedDirection, leftEyeOrigin, leftEyeDirection, rightEyeOrigin, rightEyeDirection;
	
	#ifdef WIN32
		// Transform data into Gibson coordinate system before returning to user
		glm::vec3 combinedGlmOrigin(vrToGib * glm::vec4(eyeTrackingData.combinedOrigin, 1.0));
		glm::vec3 combinedGlmDirection(vrToGib * glm::vec4(eyeTrackingData.combinedDirection, 1.0));
		glm::vec3 leftGlmOrigin(vrToGib * glm::vec4(eyeTrackingData.combinedOrigin, 1.0));
		glm::vec3 leftGlmDirection(vrToGib * glm::vec4(eyeTrackingData.combinedDirection, 1.0));
		glm::vec3 rightGlmOrigin(vrToGib * glm::vec4(eyeTrackingData.combinedOrigin, 1.0));
		glm::vec3 rightGlmDirection(vrToGib * glm::vec4(eyeTrackingData.combinedDirection, 1.0));
		
		combinedOrigin.append(combinedGlmOrigin.x);
		combinedOrigin.append(combinedGlmOrigin.y);
		combinedOrigin.append(combinedGlmOrigin.z);
		combinedDirection.append(combinedGlmDirection.x);
		combinedDirection.append(combinedGlmDirection.y);
		combinedDirection.append(combinedGlmDirection.z);

		leftEyeOrigin.append(leftGlmOrigin.x);
		leftEyeOrigin.append(leftGlmOrigin.y);
		leftEyeOrigin.append(leftGlmOrigin.z);
		leftGlmDirection.append(leftGlmDirection.x);
		leftGlmDirection.append(leftGlmDirection.y);
		leftGlmDirection.append(leftGlmDirection.z);


		rightEyeOrigin.append(rightGlmOrigin.x);
		rightEyeOrigin.append(rightGlmOrigin.y);
		rightEyeOrigin.append(rightGlmOrigin.z);
		rightEyeDirection.append(rightGlmDirection.x);
		rightEyeDirection.append(rightGlmDirection.y);
		rightEyeDirection.append(rightGlmDirection.z);
		
		// Set validity to false if eye tracking is not being used
		if (this->useEyeTracking) {
			eyeData.append(eyeTrackingData.isCombinedPoseValid);
			eyeData.append(eyeTrackingData.isLeftPoseValid);
			eyeData.append(eyeTrackingData.isRightPoseValid);

		}
		else {
			eyeData.append(false);
			eyeData.append(false);
			eyeData.append(false);
		}
		eyeData.append(combinedOrigin);
		eyeData.append(combinedDirection);
		eyeData.append(leftEyeOrigin);
		eyeData.append(leftEyeDirection);
		eyeData.append(rightEyeOrigin);
		eyeData.append(rightEyeDirection);
		eyeData.append(eyeTrackingData.leftPupilDiameter);
		eyeData.append(eyeTrackingData.rightPupilDiameter);
		eyeData.append(eyeTrackingData.leftEyeOpenness);
		eyeData.append(eyeTrackingData.rightEyeOpenness);
	// Return dummy data with false validity if eye tracking is not enabled (on non-Windows system)
	#else
		eyeData.append(false);
		eyeData.append(false);
		eyeData.append(false);
		eyeData.append(combinedOrigin);
		eyeData.append(combinedDirection);
		eyeData.append(leftEyeOrigin);
		eyeData.append(leftEyeDirection);
		eyeData.append(rightEyeOrigin);
		eyeData.append(rightEyeDirection);
		// pupil diameter and eye openess has value -1 for invalidity
		eyeData.append(-1);
		eyeData.append(-1);
		eyeData.append(-1);
		eyeData.append(-1);
	#endif
	return eyeData;
}



/// @brief Returns whether the current VR system supports eye tracking
/// @return whether eye tracking is supported
bool VRSys::hasEyeTrackingSupport() {
	#ifdef WIN32
		return ViveSR::anipal::Eye::IsViveProEye();
	#else
		return false;	// Non-windows OS always have eye tracking disabled
	#endif
}



/// @brief Get the VR offset vector
/// @return [x, y, z] in OG coordinate system
py::list VRSys::getVROffset() {
	glm::vec3 transformedOffsetVec(vrToGib * glm::vec4(this->vrOffsetVec, 1.0));

	py::list offset;
	offset.append(transformedOffsetVec.x);
	offset.append(transformedOffsetVec.y);
	offset.append(transformedOffsetVec.z);

	return offset;
}



/// @brief Set the VR offset of the VR headset
void VRSys::setVROffset(float x, float y, float z) {
	this->vrOffsetVec = gibToVr * glm::vec4(x, y, z, 1.0);
}



/// @brief Causes a haptic pulse in the specified controller, for a user-specified duration
/// @param device either left_controller or right_controller
/// @param intensity range from 0 to 1 
/// @note Haptic pulses can only trigger every 5ms, regardless of duration
/// TIMELINE: Call after physics/rendering have been stepped in the simulator
void VRSys::triggerHapticPulse(char* device, float intensity) {
	vr::TrackedDeviceIndex_t deviceIndex;
	if (!strcmp(device, "left_controller")) 		deviceIndex = leftControllerData.index;
	else if (!strcmp(device, "right_controller")) 	deviceIndex = rightControllerData.index;
	else {
		std::cerr << "[VRSys] HAPTIC ERROR: Device " << device << " must be a controller." << std::endl;
		return;
	}
	// Currently haptics are only supported on one axis (touchpad axis)
	// Multiply intensity wth 4000 microseconds
	m_pHMD->TriggerHapticPulse(deviceIndex, 0, (unsigned short) (intensity * 4000));
}



// VR overlay methods

/// @brief Create a new overlay
/// @param name name of the overlay
/// @param width width of the overlay in meters
/// @param pos_x x position of the overlay relative to the hmd
/// @param pos_y y position of teh overlay relative to the hmd
/// @param pos_z z position of the overlay relative to the hmd
/// @param fpath file path that sets the overlay image from (default is empty string)
void VRSys::createOverlay(char* name, float width, float pos_x, float pos_y, float pos_z, char* fpath) {
	vr::VROverlayHandle_t handle;
	vr::VROverlay()->CreateOverlay(name, name, &handle);
	if (strcmp(fpath, "") != 0) {
		vr::VROverlay()->SetOverlayFromFile(handle, fpath);
	}
	vr::VROverlay()->SetOverlayWidthInMeters(handle, width);
	
	vr::HmdMatrix34_t transform = {
		1.0f, 0.0f, 0.0f, pos_x,
		0.0f, 1.0f, 0.0f, pos_y,
		0.0f, 0.0f, 1.0f, pos_z
	};
	std::string ovName = std::string(name);
	this->overlayNamesToHandles[ovName] = handle;

	vr::VROverlayError overlayError = vr::VROverlay()->SetOverlayTransformTrackedDeviceRelative(handle, vr::k_unTrackedDeviceIndex_Hmd, &transform);
	if (overlayError != vr::VROverlayError_None) {
		std::cerr << "[VRSys] Unable to set overlay relative to HMD for name " << ovName << std::endl;
	}
}



/// @brief crop overlay by size (TODO: modify this)
/// @param name name of the overlay
/// @param start_u 
/// @param start_v 
/// @param end_u 
/// @param end_v 
void VRSys::cropOverlay(char* name, float start_u, float start_v, float end_u, float end_v) {
	std::string ovName(name);
	vr::VROverlayHandle_t handle = this->overlayNamesToHandles[ovName];

	// Create texture bounds and crop overlay
	vr::VRTextureBounds_t texBounds;
	texBounds.uMin = start_u;
	texBounds.vMin = start_v;
	texBounds.uMax = end_u;
	texBounds.vMax = end_v;

	vr::VROverlayError overlayError = vr::VROverlay()->SetOverlayTextureBounds(handle, &texBounds);
	if (overlayError != vr::VROverlayError_None) {
		std::cerr << "[VRSys] Unable to crop overlay with name " << ovName << std::endl;
	}
}



/// @brief destroy overlay by name
/// @param name name of the overlay
void VRSys::destroyOverlay(char* name) {
	std::string ovName(name);
	vr::VROverlayHandle_t handle = this->overlayNamesToHandles[ovName];
	vr::VROverlayError overlayError = vr::VROverlay()->DestroyOverlay(handle);
	if (overlayError != vr::VROverlayError_None) {
		std::cerr << "[VRSys] Unable to destroy overlay with name " << ovName << std::endl;
	}
}



/// @brief hide overlay by name
/// @param name name of the overlay
void VRSys::hideOverlay(char* name) {
	vr::VROverlay()->HideOverlay(this->overlayNamesToHandles[std::string(name)]);
}



/// @brief show overlay by name
/// @param name name of the overlay
void VRSys::showOverlay(char* name) {
	vr::VROverlay()->ShowOverlay(this->overlayNamesToHandles[std::string(name)]);
}



/// @brief update overlay texture with GL buffer
/// @param name name of the overlay
/// @param texID GL texture ID of the overlay
void VRSys::updateOverlayTexture(char* name, GLuint texID) {
	vr::Texture_t texture = { (void*)(uintptr_t)texID, vr::TextureType_OpenGL, vr::ColorSpace_Auto };
	vr::VROverlayError overlayError = vr::VROverlay()->SetOverlayTexture(this->overlayNamesToHandles[std::string(name)], &texture);
	if (overlayError != vr::VROverlayError_None) {
		std::cerr << "[VRSys] Unable to set texture for overlay with name " << std::string(name) << std::endl;
	}
}

# pragma endregion



# pragma region Private Methods

/// @brief Converts a SteamVR Matrix to a glm matrix
/// @param matPose 4*4 SteamVR matrix
/// @return 4*4 glm matrix
glm::mat4 VRSys::convertSteamVRMatrixToGlmMat4(const vr::HmdMatrix34_t& matPose) {
	glm::mat4 mat(
		matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
		matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
		matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
		matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
	);
	return mat;
}



/// @brief Generates a pose matrix for the specified eye (left or right)
glm::mat4 VRSys::getHMDEyePose(vr::Hmd_Eye eye) {
	vr::HmdMatrix34_t eye2Head = m_pHMD->GetEyeToHeadTransform(eye);
	return glm::inverse(convertSteamVRMatrixToGlmMat4(eye2Head));
}



/// @brief Generates a projection matrix for the specified eye (left or right)
glm::mat4 VRSys::getHMDEyeProjection(vr::Hmd_Eye eye) {
	vr::HmdMatrix44_t mat = m_pHMD->GetProjectionMatrix(eye, nearClip, farClip);

	glm::mat4 eyeProjMat(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
		mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
		mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
		mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
	);

	return eyeProjMat;
}



/// @brief Get position of device from SteamVR matrix
/// @param matrix 4*4 pose matrix
/// @return [x, y, z]
glm::vec3 VRSys::getPositionFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
	return glm::vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}



/// @brief Get rotation of device in quaternion from SteamVR matrix
/// @param matrix 4*4 pose matrix
/// @return [w, x, y, z] 
glm::vec4 VRSys::getRotationFromSteamVRMatrix(vr::HmdMatrix34_t& matrix) {
	glm::vec4 q;

	q.w = (float)sqrt(fmax(0, 1 + matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2])) / 2;
	q.x = (float)sqrt(fmax(0, 1 + matrix.m[0][0] - matrix.m[1][1] - matrix.m[2][2])) / 2;
	q.y = (float)sqrt(fmax(0, 1 - matrix.m[0][0] + matrix.m[1][1] - matrix.m[2][2])) / 2;
	q.z = (float)sqrt(fmax(0, 1 - matrix.m[0][0] - matrix.m[1][1] + matrix.m[2][2])) / 2;
	q.x = copysign(q.x, matrix.m[2][1] - matrix.m[1][2]);
	q.y = copysign(q.y, matrix.m[0][2] - matrix.m[2][0]);
	q.z = copysign(q.z, matrix.m[1][0] - matrix.m[0][1]);

	return q;
}



/// @brief Get vector 3 representation of column from glm mat 4 (useful for extracting rotation component)
/// @param col_index int representing column number
/// @param mat 4*4 pose matrix
/// @return column vector
glm::vec3 VRSys::getVec3ColFromMat4(int col_index, glm::mat4& mat) {
	glm::vec3 v;
	v.x = mat[col_index][0];
	v.y = mat[col_index][1];
	v.z = mat[col_index][2];
	return v;
}



#ifdef WIN32
/// @brief Polls SRAnipal to get updated eye tracking information
/// @note We need to convert from SRanipal coordinate system to OpenVR coordinate system
/// @ref https://forum.vive.com/topic/5888-vive-pro-eye-finding-a-single-eye-origin-in-world-space/?ct=1593593815
void VRSys::pollAnipal() {
	while (!this->shouldShutDownEyeTracking) {
		if (ViveSR::anipal::Eye::GetEyeData(&this->eyeData) == ViveSR::Error::WORK) { 
			// Record pupil measurements
			eyeTrackingData.leftPupilDiameter = -1;
			eyeTrackingData.rightPupilDiameter = -1;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY))
				eyeTrackingData.leftPupilDiameter = this->eyeData.verbose_data.left.pupil_diameter_mm;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY))
				eyeTrackingData.rightPupilDiameter = this->eyeData.verbose_data.right.pupil_diameter_mm; 
			// Record eye openness
			eyeTrackingData.leftEyeOpenness = -1;
			eyeTrackingData.rightEyeOpenness = -1;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY))
				eyeTrackingData.leftEyeOpenness = this->eyeData.verbose_data.left.eye_openness;
			if (ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY))
				eyeTrackingData.rightEyeOpenness = this->eyeData.verbose_data.right.eye_openness; 

			// Both origin and dir are relative to the HMD coordinate system, so we need to transform them into HMD coordinate system
			eyeTrackingData.isCombinedPoseValid = false;
			eyeTrackingData.isLeftPoseValid = false;
			eyeTrackingData.isRightPoseValid = false;
			if (!hmdData.isValidData) {
				continue;
			}
			// combined eye origin and direction
			int isCombinedOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isCombinedDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.combined.eye_data.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isCombinedOriginValid && isCombinedDirValid) {
				eyeTrackingData.isCombinedPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto combinedGazeOrigin = this->eyeData.verbose_data.combined.eye_data.gaze_origin_mm;
				glm::vec3 combinedEyeSpaceOrigin(-1 * combinedGazeOrigin.x / 1000.0f, combinedGazeOrigin.y / 1000.0f, -1 * combinedGazeOrigin.z / 1000.0f);
				eyeTrackingData.combinedOrigin = glm::vec3(hmdData.deviceTransform * glm::vec4(combinedEyeSpaceOrigin, 1.0));

				auto combinedGazeDirection = this->eyeData.verbose_data.combined.eye_data.gaze_direction_normalized;
				// Convert to OpenVR coordinates
				glm::vec3 combinedEyeSpaceDir(-1 * combinedGazeDirection.x, combinedGazeDirection.y, -1 * combinedGazeDirection.z);
				// Only rotate, no translate - remove translation to preserve rotation
				glm::vec3 combinedHmdSpaceDir(hmdData.deviceTransform * glm::vec4(combinedEyeSpaceDir, 1.0));
				// Make sure to normalize (and also flip x and z, since anipal coordinate convention is different to OpenGL)
				eyeTrackingData.combinedDirection = glm::normalize(glm::vec3(combinedHmdSpaceDir.x - hmdData.devicePos.x, combinedHmdSpaceDir.y - hmdData.devicePos.y, combinedHmdSpaceDir.z - hmdData.devicePos.z));
			}

			// left eye origin and direction
			int isLeftOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isLeftDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.left.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isLeftOriginValid && isLeftDirValid) {
				eyeTrackingData.isLeftPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto leftGazeOrigin = this->eyeData.verbose_data.left.gaze_origin_mm;
				glm::vec3 leftEyeSpaceOrigin(-1 * leftGazeOrigin.x / 1000.0f, leftGazeOrigin.y / 1000.0f, -1 * leftGazeOrigin.z / 1000.0f);
				eyeTrackingData.leftOrigin = glm::vec3(hmdData.deviceTransform * glm::vec4(leftEyeSpaceOrigin, 1.0));

				auto leftGazeDirection = this->eyeData.verbose_data.left.gaze_direction_normalized;
				// Convert to OpenVR coordinates
				glm::vec3 leftEyeSpaceDir(-1 * leftGazeDirection.x, leftGazeDirection.y, -1 * leftGazeDirection.z);
				// Only rotate, no translate - remove translation to preserve rotation
				glm::vec3 leftHmdSpaceDir(hmdData.deviceTransform * glm::vec4(leftEyeSpaceDir, 1.0));
				// Make sure to normalize (and also flip x and z, since anipal coordinate convention is different to OpenGL)
				eyeTrackingData.leftDirection = glm::normalize(glm::vec3(leftHmdSpaceDir.x - hmdData.devicePos.x, leftHmdSpaceDir.y - hmdData.devicePos.y, leftHmdSpaceDir.z - hmdData.devicePos.z));
			}

			// right eye origin and direction
			int isRightOriginValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY);
			int isRightDirValid = ViveSR::anipal::Eye::DecodeBitMask(this->eyeData.verbose_data.right.eye_data_validata_bit_mask,
				ViveSR::anipal::Eye::SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY);
			if (isRightOriginValid && isRightDirValid) {
				eyeTrackingData.isRightPoseValid = true;
				// Returns value in mm, so need to divide by 1000 to get meters (OG uses meters)
				auto rightGazeOrigin = this->eyeData.verbose_data.right.gaze_origin_mm;
				glm::vec3 rightEyeSpaceOrigin(-1 * rightGazeOrigin.x / 1000.0f, rightGazeOrigin.y / 1000.0f, -1 * rightGazeOrigin.z / 1000.0f);
				eyeTrackingData.rightOrigin = glm::vec3(hmdData.deviceTransform * glm::vec4(rightEyeSpaceOrigin, 1.0));

				auto rightGazeDirection = this->eyeData.verbose_data.right.gaze_direction_normalized;
				// Convert to OpenVR coordinates
				glm::vec3 rightEyeSpaceDir(-1 * rightGazeDirection.x, rightGazeDirection.y, -1 * rightGazeDirection.z);
				// Only rotate, no translate - remove translation to preserve rotation
				glm::vec3 rightHmdSpaceDir(hmdData.deviceTransform * glm::vec4(rightEyeSpaceDir, 1.0));
				// Make sure to normalize (and also flip x and z, since anipal coordinate convention is different to OpenGL)
				eyeTrackingData.rightDirection = glm::normalize(glm::vec3(rightHmdSpaceDir.x - hmdData.devicePos.x, rightHmdSpaceDir.y - hmdData.devicePos.y, rightHmdSpaceDir.z - hmdData.devicePos.z));
			}
		}
	}
}
#endif



/// @brief Print string version of mat4
void VRSys::printMat4(glm::mat4& m) {
	printf("%s", glm::to_string(m).c_str());
	printf("\n");
}



/// @brief Print string version of vec3
void VRSys::printVec3(glm::vec3& v) {
	printf("%s", glm::to_string(v).c_str());
	printf("\n");
}



/// @brief Sets the position component of a SteamVR Matrix
/// @param pos glm::vec3 position vectorpupil_diameter_mm
/// @param mat 4*4 SteamVR pose matrix
void VRSys::setSteamVRMatrixPosition(glm::vec3& pos, vr::HmdMatrix34_t& mat) {
	mat.m[0][3] = pos[0];
	mat.m[1][3] = pos[1];
	mat.m[2][3] = pos[2];
}

# pragma endregion



PYBIND11_MODULE(VRSys, m) {

	py::class_<VRSys> pymodule = py::class_<VRSys>(m, "VRSys");

	pymodule.def(py::init<>());
	// VR functions
	pymodule.def("init", 						&VRSys::init);
	pymodule.def("release", 					&VRSys::release);
	pymodule.def("preRender", 					&VRSys::preRender);
	pymodule.def("render", 						&VRSys::render);
	pymodule.def("pollVREvents", 				&VRSys::pollVREvents);
	pymodule.def("pollVRPosesAndStates", 		&VRSys::pollVRPosesAndStates);
	pymodule.def("getControllerButtonData", 	&VRSys::getControllerButtonData);
	pymodule.def("getDevicePose", 				&VRSys::getDevicePose);
	pymodule.def("getDeviceCoordinateSystem", 	&VRSys::getDeviceCoordinateSystem);
	pymodule.def("getEyeTrackingData", 			&VRSys::getEyeTrackingData);
	pymodule.def("hasEyeTrackingSupport", 		&VRSys::hasEyeTrackingSupport);
	pymodule.def("getVROffset", 				&VRSys::getVROffset);
	pymodule.def("setVROffset", 				&VRSys::setVROffset);
	pymodule.def("triggerHapticPulse", 			&VRSys::triggerHapticPulse);
	// VR overlay methods
	pymodule.def("createOverlay", 				&VRSys::createOverlay);
	pymodule.def("cropOverlay", 				&VRSys::cropOverlay);
	pymodule.def("destroyOverlay", 				&VRSys::destroyOverlay);
	pymodule.def("hideOverlay", 				&VRSys::hideOverlay);
	pymodule.def("showOverlay", 				&VRSys::showOverlay);
	pymodule.def("updateOverlayTexture", 		&VRSys::updateOverlayTexture);

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}