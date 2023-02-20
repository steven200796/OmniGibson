cmake_minimum_required(VERSION 3.17)
project(VRSys)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR deps)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.3
)
FetchContent_Declare(
    openvr
    GIT_REPOSITORY https://github.com/ValveSoftware/openvr.git
    GIT_TAG 1a0ea26
)

FetchContent_MakeAvailable(glfw glm pybind11 openvr)

include_directories(
    "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/glfw-src/include",
    "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/glm-src",
    "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/openvr-src/headers",
    "${FETCHCONTENT_BASE_DIR}/glad"
)
find_package(OpenGL)


if(WIN32)
    set(PATH_SUFFIXES win64 Win64 x64)
    # Find SRAnipal
    set(SRANI_DIR sranipal)
    find_library(
        SRANI_LIBRARIES
        NAMES SRAnipal
        PATHS "${SRANI_DIR}/lib"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    include_directories("${SRANI_DIR}/include")
else()
    set(PATH_SUFFIXES linux64 Linux64 x64)
endif()


find_library(
    OPENVR_LIBRARIES
    NAMES openvr_api
    PATHS "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/openvr-src/bin" "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/openvr-src/lib"
    PATH_SUFFIXES ${PATH_SUFFIXES}
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
)
    
add_library(VRSys MODULE deps/glad/glad.c cpp/vrsys.cpp)
if(WIN32)
    target_link_libraries(VRSys
        PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OPENVR_LIBRARIES} ${SRANI_LIBRARIES}
    )
else()
    target_link_libraries(VRSys
        PRIVATE pybind11::module ${CMAKE_DL_LIBS} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${OPENVR_LIBRARIES}
    )
endif()

set_target_properties(VRSys PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")