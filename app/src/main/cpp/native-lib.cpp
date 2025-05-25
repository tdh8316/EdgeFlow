#include "edgeflow/EdgeFlow.h"
#include <android/log.h>
#include <arm_compute/runtime/Tensor.h>
#include <jni.h>
#include <string>

#define MainActivity(func) Java_app_edgeflow_MainActivity_##func

/// EdgeFlow singleton instance
static EdgeFlow &g_edgeflow = EdgeFlow::instance();

extern "C" JNIEXPORT jstring JNICALL
MainActivity(stringFromJNI)(JNIEnv *env, jobject /* this */) {
  std::string hello = "Hello from C++ backend!";
  return env->NewStringUTF(hello.c_str());
}

/// JNI function to initialize EdgeFlow
/// @param env
/// @param
/// @param model_dag_path The path to the model DAG file (JSON format)
/// @param device_info The local device information (JSON string)
/// @param devices The list of devices to be used (JSON string)
extern "C" JNIEXPORT jboolean JNICALL
MainActivity(initializeEdgeFlow)(
        JNIEnv *env,
        jobject /* this */,
        jstring model_dag_path_jstr,
        jstring device_info_jstr,
        jstring devices_jstr) {
  /* Convert jstring to std::string */
  const char *model_dag_path_c = env->GetStringUTFChars(model_dag_path_jstr, nullptr);
  const auto model_dag_path_str = std::string(model_dag_path_c);

  const char *device_info_c = env->GetStringUTFChars(device_info_jstr, nullptr);
  const auto device_info_str = std::string(device_info_c);

  const char *devices_c = env->GetStringUTFChars(devices_jstr, nullptr);
  const auto devices_str = std::string(devices_c);

  /* Create a ModelDAG object */
  auto dag = std::make_shared<ModelDAG>();
  // TODO: Load the model DAG from the JSON file

  // Sample model: simple MLP for XOR operation
  {
    dag->name = "SimpleXOR";
    dag->input_shape = arm_compute::TensorShape(2);
    dag->output_shape = arm_compute::TensorShape(1);

    // Weights and biases for the first layer
    auto weights0 = std::make_unique<arm_compute::Tensor>();
    auto bias0 = std::make_unique<arm_compute::Tensor>();
    {
      weights0->allocator()->init(
              // TensorInfo shape for weights: (in_feature, out_feature)
              arm_compute::TensorInfo({2, 2}, 1, arm_compute::DataType::F32));
      weights0->allocator()->allocate();
      float weights0_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      std::memcpy(weights0->buffer(), weights0_data, sizeof(weights0_data));

      bias0->allocator()->init(
              // TensorInfo shape for bias: (out_feature,)
              arm_compute::TensorInfo(2, 1, arm_compute::DataType::F32));
      bias0->allocator()->allocate();
      float bias0_data[2] = {0.0f, -1.0f};
      std::memcpy(bias0->buffer(), bias0_data, sizeof(bias0_data));
    }

    // Weights and biases for the second layer
    auto weights1 = std::make_unique<arm_compute::Tensor>();
    auto bias1 = std::make_unique<arm_compute::Tensor>();
    {
      weights1->allocator()->init(
              // TensorInfo shape for weights: (in_feature, out_feature)
              arm_compute::TensorInfo({2, 1}, 1, arm_compute::DataType::F32));
      weights1->allocator()->allocate();
      float weights1_data[2] = {1.0f, -2.0f};
      std::memcpy(weights1->buffer(), weights1_data, sizeof(weights1_data));

      bias1->allocator()->init(
              // TensorInfo shape for bias: (out_feature,)
              arm_compute::TensorInfo(1, 1, arm_compute::DataType::F32));
      bias1->allocator()->allocate();
      float bias1_data[1] = {0.0f};
      std::memcpy(bias1->buffer(), bias1_data, sizeof(bias1_data));
    }

    // Layer parameters
    auto layer0_params = std::make_shared<OperatorParams>(
            LinearParams{
                    .weight = std::move(weights0),
                    .bias = std::move(bias0),
            });
    auto layer1_params = std::make_shared<OperatorParams>(
            LinearParams{
                    .weight = std::move(weights1),
                    .bias = std::move(bias1),
            });

    // Define the layers
    {
      dag->layers["layer0"] = Layer{.id = "layer0",
                                    .type = OperatorType::Linear,
                                    .params = layer0_params,
                                    .input_shape = arm_compute::TensorShape(2),
                                    .output_shape = arm_compute::TensorShape(2)};
      dag->layers["act0"] = Layer{.id = "act0",
                                  .type = OperatorType::Activation,
                                  .params = std::make_shared<OperatorParams>(
                                          ActivationParams{
                                                  .type = ActivationType::ReLU,
                                          }),
                                  .input_shape = arm_compute::TensorShape(2),
                                  .output_shape = arm_compute::TensorShape(2)};
      dag->layers["layer1"] = Layer{.id = "layer1",
                                    .type = OperatorType::Linear,
                                    .params = layer1_params,
                                    .input_shape = arm_compute::TensorShape(2),
                                    .output_shape = arm_compute::TensorShape(1)};
      dag->layers["act1"] = Layer{.id = "act1",
                                  .type = OperatorType::Activation,
                                  .params = std::make_shared<OperatorParams>(
                                          ActivationParams{
                                                  .type = ActivationType::ReLU,
                                          }),
                                  .input_shape = arm_compute::TensorShape(1),
                                  .output_shape = arm_compute::TensorShape(1)};
    }


    // Define the execution units for each layer
    {
      dag->eus["layer0::eu0"] = ExecutionUnit{
              .layer_id = "layer0",
              .device_id = "device0",
              .id = "layer0::eu0",
              .input_requirements = {}, // No input requirements for the first layer
              .output_range = Range{0, 2},
              .op = Operator{
                      .type = OperatorType::Linear,
                      .params = dag->layers["layer0"].params,
              },
              .forward_table = {.entries{
                      {
                              "act0::eu0",
                              Range{0, 2},
                      },
              }},
              .expected_input_shape = arm_compute::TensorShape(2),
              .expected_output_shape = arm_compute::TensorShape(2),
              .is_leaf = false,
              .is_root = true,
      };
      dag->eus["act0::eu0"] = ExecutionUnit{
              .layer_id = "act0",
              .device_id = "device0",
              .id = "act0::eu0",
              .input_requirements = {},
              .output_range = Range{0, 2},
              .op = Operator{
                      .type = OperatorType::Activation,
                      .params = dag->layers["act0"].params,
              },
              .forward_table = {.entries{
                      {
                              "layer1::eu0",
                              Range{0, 2},
                      },
              }},
              .expected_input_shape = arm_compute::TensorShape(2),
              .expected_output_shape = arm_compute::TensorShape(2),
              .is_leaf = false,
              .is_root = false,
      };
      dag->eus["layer1::eu0"] = ExecutionUnit{
              .layer_id = "layer1",
              .device_id = "device0",
              .id = "layer1::eu0",
              .input_requirements = {},
              .output_range = Range{0, 1},
              .op = Operator{
                      .type = OperatorType::Linear,
                      .params = dag->layers["layer1"].params,
              },
              .forward_table = {.entries{
                      {
                              "act1::eu0",
                              Range{0, 1},
                      },
              }},
              .expected_input_shape = arm_compute::TensorShape(2),
              .expected_output_shape = arm_compute::TensorShape(1),
              .is_leaf = false,
              .is_root = false,
      };
      dag->eus["act1::eu0"] = ExecutionUnit{
              .layer_id = "act1",
              .device_id = "device0",
              .id = "act1::eu0",
              .input_requirements = {},
              .output_range = Range{0, 1},
              .op = Operator{
                      .type = OperatorType::Activation,
                      .params = dag->layers["act1"].params,
              },
              .forward_table = {}, // No forwarding table for the last layer
              .expected_input_shape = arm_compute::TensorShape(1),
              .expected_output_shape = arm_compute::TensorShape(1),
              .is_leaf = true,
              .is_root = false,
      };
    }
  }

  /* Create a DeviceInfo object */
  auto device_info = std::make_shared<DeviceInfo>();
  device_info->id = "device0"; // TODO: Parse this from the device_info_str JSON string

  /* Create a device list */
  std::vector<DeviceInfo> devices_list{};
  // TODO: Parse the devices_str JSON string

  /* Initialize EdgeFlow */
  bool result = g_edgeflow.initialize(dag, device_info, devices_list);
  if (!result) {
    __android_log_print(ANDROID_LOG_ERROR, "initializeEdgeFlow",
                        "Failed to initialize EdgeFlow");
  }

  /* Release the jstring */
  env->ReleaseStringUTFChars(model_dag_path_jstr, model_dag_path_c);
  env->ReleaseStringUTFChars(device_info_jstr, device_info_c);
  env->ReleaseStringUTFChars(devices_jstr, devices_c);

  return result ? JNI_TRUE : JNI_FALSE;
}

/// JNI function to start inference
/// @param env
/// @param
/// @param input The input tensor (JSON string)
extern "C" JNIEXPORT jboolean JNICALL
MainActivity(startInference)(
        JNIEnv *env,
        jobject /* this */,
        jstring input) {
  /* Convert jstring to std::string */
  const char *input_c = env->GetStringUTFChars(input, nullptr);
  const auto input_str = std::string(input_c);

  /* Create an input tensor */
  auto input_tensor = std::make_unique<arm_compute::Tensor>();
  // TODO: Parse the input JSON string and fill the tensor
  input_tensor->allocator()->init(
          arm_compute::TensorInfo(2, 1, arm_compute::DataType::F32));
  input_tensor->allocator()->allocate();
  float test_data[2];
  // TODO: Parse the input_str array

  // For now, assume the input is a simple array of floats e.g., "1,2,..."
  {
    std::istringstream iss(input_str);
    std::string token;
    size_t i = 0;
    while (std::getline(iss, token, ',')) {
      if (i < 2) {
        test_data[i] = std::stof(token);
        ++i;
      } else {
        break;
      }
    }
    if (i < 2) {
      __android_log_print(ANDROID_LOG_ERROR, "startInference",
                          "Invalid input data: %s", input_str.c_str());
      env->ReleaseStringUTFChars(input, input_c);
      return JNI_FALSE;
    }

    memcpy(input_tensor->buffer(), test_data, sizeof(test_data));
    __android_log_print(ANDROID_LOG_INFO, "startInference",
                        "Input tensor allocated with data: %f, %f",
                        test_data[0], test_data[1]);
  }

  /* Release the jstring */
  env->ReleaseStringUTFChars(input, input_c);

  return g_edgeflow.inference(std::move(input_tensor)) ? JNI_TRUE : JNI_FALSE;
}

/// JNI function to register the JNI callback that will be invoked
/// when the inference is completed by the EdgeFlow instance.
/// @param env
/// @param
/// @param j_obj The Java object to call the callback on
extern "C" JNIEXPORT jboolean JNICALL
MainActivity(registerJavaCallback)(
        JNIEnv *env,
        jobject /* this */,
        jobject thiz) {
  jclass cls = env->GetObjectClass(thiz);
  if (cls == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, "registerJavaCallback",
                        "Failed to get Java class reference");
    return false;
  }

  // Obtain the method ID of `onInferenceComplete` method in Java
  jmethodID method = env->GetMethodID(cls, "onInferenceComplete", "(Ljava/lang/String;)V");
  if (method == nullptr) {
    __android_log_print(ANDROID_LOG_ERROR, "registerJavaCallback",
                        "Failed to get method ID");
    env->DeleteLocalRef(cls);
    return false;
  }
  env->DeleteLocalRef(cls);

  // Register the callback with EdgeFlow
  g_edgeflow.register_jni_callback(env, thiz, method);
  __android_log_print(ANDROID_LOG_INFO, "registerJavaCallback",
                      "Java callback registered successfully with EdgeFlow C++ backend");
  return true;
}
