#ifndef VR_SYS_HEADER
#define VR_SYS_HEADER

#ifdef WIN32
#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Enums.h"
#endif

#include <thread>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <map>
#include <openvr.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>

namespace py = pybind11;


class VRSys {
public:
	GLFWwindow *window = NULL;
	// Pointer used to reference VR system
	vr::IVRSystem* m_pHMD;
	float nearClip;
	float farClip;

	// Vector indicating the user-defined offset for the VR system (may be used if implementing a teleportation movement scheme, for example)
	glm::vec3 vrOffsetVec;

	// Device data stored in VR coordinates
	struct DeviceData {
		glm::mat4 deviceTransform;		// standard 4x4 transform
		glm::vec3 devicePos;
		glm::vec4 deviceRot;			// w, x, y, z (quaternion)
		bool isValidData = false;		// is device valid and being tracked
		int index = -1;					// index of current device in device array
		float triggerFraction;				// [Controllers only] trigger pressed fraction (0 min, 1 max)
		glm::vec2 touchpadAnalogVector;	// [Controllers only] analog touch vector
		uint64_t buttonsPressed;		// the buttons that are currently pressed, as a bit vector. bit at position i is state of button i.
	};

	DeviceData hmdData;
	DeviceData leftControllerData;
	DeviceData rightControllerData;

	// Indicates the position of hmd in the room (without offset)
	glm::vec3 hmdActualPosition;

	// Matrices for both left and right eyes (only proj and view are actually returned to the user)
	glm::mat4 leftEyeProj;
	glm::mat4 leftEyePose;
	glm::mat4 leftEyeView;
	glm::mat4 rightEyeProj;
	glm::mat4 rightEyePose;
	glm::mat4 rightEyeView;

	// Matrices that can transform between OmniGibson and VR coordinate system
	glm::mat4 gibToVr;
	glm::mat4 vrToGib;

	bool useEyeTracking;
	bool shouldShutDownEyeTracking;

	#ifdef WIN32
		// SRAnipal variables
		std::thread* eyeTrackingThread;
		ViveSR::anipal::Eye::EyeData eyeData;
	#endif

	// Struct storing eye data for SR anipal
	struct EyeTrackingData {
		bool isCombinedPoseValid = false;
		bool isLeftPoseValid = false;
		bool isRightPoseValid = false;
		// All in world space
		glm::vec3 combinedOrigin;
		glm::vec3 combinedDirection;
		glm::vec3 leftEyeOrigin;
		glm::vec3 leftEyeDirection;
		glm::vec3 rightEyeOrigin;
		glm::vec3 rightEyeDirection;
		// Both in mm
		float leftPupilDiameter;
		float rightPupilDiameter;
		// Both in 0~1
		float leftEyeOpenness;
		float rightEyeOpenness;
	};
	EyeTrackingData eyeTrackingData;

	// Stores mapping from overlay names to handles
	std::map<std::string, vr::VROverlayHandle_t> overlayNamesToHandles;

	// Stores mapping from tracker serial names to device data
	std::map<std::string, DeviceData> trackerNamesToData;


	// Main VR methods

	VRSys(): m_pHMD(NULL) {};
	
	void init(bool useEyeTracking);

	void release();

	py::list preRender();

	void render(GLuint leftTexID, GLuint rightTexID);

	py::list pollVREvents();

	void pollVRPosesAndStates();

	py::list getControllerButtonData(char* controllerType);

	py::list getDevicePose(char* deviceType);

	py::list getDeviceCoordinateSystem(char* device);

	py::list getEyeTrackingData();

	bool hasEyeTrackingSupport();

	py::list getVROffset();
	
	void setVROffset(float x, float y, float z);

	void triggerHapticPulse(char* device, float intensity);

	// VR Overlay methods

	void createOverlay(char* name, float width, float pos_x, float pos_y, float pos_z, char* fpath);

	void cropOverlay(char* name, float start_u, float start_v, float end_u, float end_v);

	void destroyOverlay(char* name);

	void hideOverlay(char* name);

	void showOverlay(char* name);

	void updateOverlayTexture(char* name, GLuint texID);

private:
	glm::mat4 convertSteamVRMatrixToGlmMat4(const vr::HmdMatrix34_t& matPose);

	glm::mat4 getHMDEyePose(vr::Hmd_Eye eye);

	glm::mat4 getHMDEyeProjection(vr::Hmd_Eye eye);

	glm::vec3 getPositionFromSteamVRMatrix(vr::HmdMatrix34_t& matrix);

	glm::vec4 getRotationFromSteamVRMatrix(vr::HmdMatrix34_t& matrix);

	glm::vec3 getVec3ColFromMat4(int col_index, glm::mat4& mat);

	void pollAnipal();

	void printMat4(glm::mat4& m);

	void printVec3(glm::vec3& v);

	void setSteamVRMatrixPosition(glm::vec3& pos, vr::HmdMatrix34_t& mat);
};

#endif