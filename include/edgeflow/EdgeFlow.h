#ifndef EDGEFLOW_EDGEFLOW_H
#define EDGEFLOW_EDGEFLOW_H

#include "edgeflow/ComputationEngine.h"
#include "edgeflow/DataTypes.h"
#include "edgeflow/NetworkEventHandler.h"
#include "edgeflow/Orchestrator.h"
#include <jni.h>

/// EdgeFlow is the main class that manages the
/// distributed inference process
class EdgeFlow {
public:
  /// Get the singleton instance of EdgeFlow
  /// @return The EdgeFlow instance
  static EdgeFlow &instance() {
    static EdgeFlow instance_;
    return instance_;
  }

  EdgeFlow(const EdgeFlow &) = delete;
  EdgeFlow &operator=(const EdgeFlow &) = delete;
  EdgeFlow(EdgeFlow &&) = delete;
  EdgeFlow &operator=(EdgeFlow &&) = delete;

  /// Initialize the EdgeFlow instance
  /// @param dag The model DAG to be executed
  /// @param device_info Local device information
  /// @param devices List of devices to be used
  bool initialize(std::unique_ptr<ModelDAG> dag,
                  std::unique_ptr<DeviceInfo> device_info,
                  const std::vector<DeviceInfo> &devices);

  /// Register the JNI completion callback for the Java side
  /// @param env
  /// @param thiz
  /// @param callback
  void register_jni_callback(JNIEnv *env, jobject thiz, jmethodID callback);

  /// Start inference using the given input tensor on the model DAG
  /// @param input The input tensor
  bool inference(std::unique_ptr<arm_compute::Tensor> input);

  /// Callback function to be called by Orchestrator
  /// when the inference process is complete.
  /// This function will invoke the registered JNI callback
  /// @param output The output tensor
  void on_inference_complete(const arm_compute::Tensor &output);

private:
  EdgeFlow() = default;

  // Model definition
  std::unique_ptr<ModelDAG> dag_ = nullptr;

  // Local device information
  std::unique_ptr<DeviceInfo> device_info_ = nullptr;

  // DeviceID |-> DeviceInfo mapping
  std::unique_ptr<DeviceMap> device_map_ = nullptr;

  // Orchestrator instance that manages the inference process
  std::unique_ptr<Orchestrator> orch_ = nullptr;

  // Manages the inference process state
  std::mutex inference_state_mtx_{};
  bool inference_active_ = false;

  /* JNI stuff */
  JavaVM *java_vm_ = nullptr;
  jobject java_callback_obj_ = nullptr;
  jmethodID java_callback_method_ = nullptr;

  bool is_initialized_ = false;
};

#endif // EDGEFLOW_EDGEFLOW_H
