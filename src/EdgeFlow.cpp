#include "edgeflow/EdgeFlow.h"
#include "edgeflow/ComputationEngine.h"
#include <android/log.h>

#include <utility>

void print_tensor(const arm_compute::Tensor &tensor, const std::string &name) {
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());
  const int n_elems =
          (int) (tensor.info()->total_size() / tensor.info()->element_size());
  std::string output_str = name + " (Shape: ";
  for (unsigned int i = 0; i < tensor.info()->num_dimensions(); ++i) {
    output_str += std::to_string(tensor.info()->dimension(i)) +
                  (i < tensor.info()->num_dimensions() - 1 ? "x" : "");
  }
  output_str += "): [";

  const auto *ptr = reinterpret_cast<float *>(tensor.buffer());
  // Print up to first 10 elements for brevity
  for (size_t i = 0; i < std::min(10, n_elems); ++i) {
    output_str += std::to_string(ptr[i]);
    if (i < n_elems - 1) {
      output_str += ", ";
    }
  }
  if (n_elems > 10) {
    output_str += "...";
  }
  output_str += "]";
  __android_log_print(ANDROID_LOG_INFO, "print_tensor", "%s",
                      output_str.c_str());
}

bool EdgeFlow::initialize(std::shared_ptr<ModelDAG> dag,
                          std::shared_ptr<DeviceInfo> device_info,
                          const std::vector<DeviceInfo> &devices) {
  if (is_initialized_) {
    __android_log_print(
            ANDROID_LOG_ERROR, "EdgeFlow::initialize",
            "EdgeFlow is already initialized. Try re-initializing.");
    delete orch_;
    orch_ = nullptr;
  }

  /* Store the model DAG and device information */
  // TODO: Validate the model DAG and device information
  dag_ = std::move(dag);
  device_info_ = std::move(device_info);
  device_map_ = std::make_shared<DeviceMap>();
  for (const auto &device: devices) {
    device_map_->emplace(device.id, device);
  }

  orch_ = new Orchestrator(dag_, device_info_, device_map_);
  orch_->register_inference_complete_callback(
          [&](const arm_compute::Tensor &output) -> void {
            on_inference_complete(output);
          });

  is_initialized_ = true;
  __android_log_print(ANDROID_LOG_INFO, "EdgeFlow::initialize",
                      "EdgeFlow initialized successfully on device: %s",
                      device_info_->id.c_str());

  return true;
}

void EdgeFlow::register_jni_callback(JNIEnv *env, jobject thiz,
                                     jmethodID callback) {
  if (!is_initialized_) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::register_jni_callback",
                        "EdgeFlow is not initialized");
    return;
  }

  env->GetJavaVM(&java_vm_);
  if (java_callback_obj_ != nullptr) {
    // JNIEnv for current thread needed for DeleteGlobalRef
    JNIEnv *current_env = nullptr;
    bool detach_needed = false;
    if (java_vm_->GetEnv(reinterpret_cast<void **>(current_env),
                         JNI_VERSION_1_6) == JNI_EDETACHED) {
      if (java_vm_->AttachCurrentThread(&current_env, nullptr) != JNI_OK) {
        __android_log_print(
                ANDROID_LOG_ERROR, "EdgeFlow::register_jni_callback",
                "Failed to attach current thread to VM for DeleteGlobalRef");
        // Proceed with caution, might leak java_callback_obj_
      } else {
        detach_needed = true;
      }
    }
    if (current_env) {
      current_env->DeleteGlobalRef(java_callback_obj_);
    }
    if (detach_needed)
      java_vm_->DetachCurrentThread();
  }
  java_callback_obj_ = env->NewGlobalRef(thiz);
  java_callback_method_ = callback;
  __android_log_print(ANDROID_LOG_INFO, "EdgeFlow::register_jni_callback",
                      "JNI callback registered successfully");
}

bool EdgeFlow::inference(std::unique_ptr<arm_compute::Tensor> input) {
  if (!is_initialized_) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::inference",
                        "EdgeFlow is not initialized");
    return false;
  }
  print_tensor(*input, "Input tensor");
  {
    std::lock_guard<std::mutex> lock(inference_state_mtx_);
    if (inference_active_) {
      __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::inference",
                          "Inference is already in progress");
      return false;
    }
    inference_active_ = true;
  }

  // Start the inference process
  if (!orch_->start_inference(std::move(input))) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::inference",
                        "Failed to start inference");
    std::lock_guard<std::mutex> lock(inference_state_mtx_);
    inference_active_ = false;
    return false;
  }

  __android_log_print(ANDROID_LOG_INFO, "EdgeFlow::inference",
                      "Inference started successfully");
  return true;
}

void EdgeFlow::on_inference_complete(const arm_compute::Tensor &output) {
  if (java_callback_obj_ == nullptr || java_callback_method_ == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::on_inference_complete",
                        "JNI callback is not registered");
    return;
  }

  JNIEnv *env;
  int getEnvStat = java_vm_->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);

  if (getEnvStat == JNI_EDETACHED) {
    __android_log_print(ANDROID_LOG_INFO, "EdgeFlow::on_inference_complete",
                        "Attaching current thread to JVM.");
    if (java_vm_->AttachCurrentThread(&env, nullptr) != 0) {
      __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::on_inference_complete",
                          "Failed to attach current thread to JVM.");
      return;
    }
  } else if (getEnvStat != JNI_OK) {
    __android_log_print(ANDROID_LOG_ERROR, "EdgeFlow::on_inference_complete",
                        "Failed to get JNI environment, status: %d",
                        getEnvStat);
    return;
  }

  // Convert the output tensor to a string representation
  // std::string output_str = "Output tensor: ";
  std::string output_str{};
  const auto *ptr = reinterpret_cast<float *>(output.buffer());
  const int n_elems =
          (int) (output.info()->total_size() / output.info()->element_size());
  for (size_t i = 0; i < n_elems; ++i) {
    output_str += std::to_string(ptr[i]);
    if (i < n_elems - 1) {
      output_str += ", ";
    }
  }

  jstring j_output_str = env->NewStringUTF(output_str.c_str());
  env->CallVoidMethod(java_callback_obj_, java_callback_method_, j_output_str);
  env->DeleteLocalRef(j_output_str);

  {
    std::lock_guard<std::mutex> lock(inference_state_mtx_);
    inference_active_ = false;
    __android_log_print(ANDROID_LOG_INFO, "EdgeFlow::on_inference_complete",
                        "Inference completed successfully");
    print_tensor(output, "EdgeFlow::on_inference_complete::output");
  }
}
