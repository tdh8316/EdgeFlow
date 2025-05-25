cmake_minimum_required(VERSION 3.22.1)

project("edgeflow")

# ARM Compute Library
set(ARM_COMPUTE_LIBRARY_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/arm_compute-v52.0.0-android-aarch64-cpu-gpu-bin")
include_directories("${ARM_COMPUTE_LIBRARY_DIR}/include")
include_directories("${ARM_COMPUTE_LIBRARY_DIR}")
add_library(arm_compute SHARED IMPORTED)
set_target_properties(arm_compute PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libarm_compute.so"
        IMPORTED_NO_SONAME 1
)

# EdgeFlow C++ implementation
set(EDGEFLOW_SRC_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src")
set(EDGEFLOW_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
include_directories("${EDGEFLOW_SRC_DIR}")
include_directories("${EDGEFLOW_INCLUDE_DIR}")

# Add the source files for the EdgeFlow library
set(EDGEFLOW_SRC_FILES
        "${EDGEFLOW_SRC_DIR}/EdgeFlow.cpp"
        "${EDGEFLOW_SRC_DIR}/ComputationEngine.cpp"
        "${EDGEFLOW_SRC_DIR}/Orchestrator.cpp"
        "${EDGEFLOW_SRC_DIR}/NetworkEventHandler.cpp"
)

set(EDGEFLOW_INCLUDE_FILES
        "${EDGEFLOW_INCLUDE_DIR}/edgeflow/DataTypes.h"
)

# Build the shared library for Android
add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        ${EDGEFLOW_SRC_FILES}
        ${EDGEFLOW_INCLUDE_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
        android
        arm_compute
        log)
