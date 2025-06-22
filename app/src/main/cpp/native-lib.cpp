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

static std::unique_ptr<ModelDAG> load_model_dag(const std::string &model_dag_path) {
  std::unique_ptr<ModelDAG> dag = std::make_unique<ModelDAG>();

  /* TODO: Load the model DAG from the JSON file
   * This function should read the JSON file at `model_dag_path`
   * and populate the `dag` object with the model structure.
   * For now, we return a sample model DAG.
   */
  // Sample model: simple MLP for XOR operation
  dag->name = "SimpleXOR";
  dag->input_shape = arm_compute::TensorShape(2);
  dag->output_shape = arm_compute::TensorShape(1);

  /* == Define layers and their parameters == */
  float weights0_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float bias0_data[2] = {0.0f, -1.0f};
  float weights1_data[2] = {1.0f, -2.0f};
  float bias1_data[1] = {0.0f};

  auto weights0 = std::make_unique<arm_compute::Tensor>();
  auto bias0 = std::make_unique<arm_compute::Tensor>();
  weights0->allocator()->init(
      // TensorInfo shape for weights: (in_feature, out_feature)
      arm_compute::TensorInfo({2, 2}, 1, arm_compute::DataType::F32));
  weights0->allocator()->allocate();
  bias0->allocator()->init(
      // TensorInfo shape for bias: (out_feature,)
      arm_compute::TensorInfo(2, 1, arm_compute::DataType::F32));
  bias0->allocator()->allocate();

  auto weights1 = std::make_unique<arm_compute::Tensor>();
  auto bias1 = std::make_unique<arm_compute::Tensor>();
  weights1->allocator()->init(
      // TensorInfo shape for weights: (in_feature, out_feature)
      arm_compute::TensorInfo({2, 1}, 1, arm_compute::DataType::F32));
  weights1->allocator()->allocate();
  bias1->allocator()->init(
      // TensorInfo shape for bias: (out_feature,)
      arm_compute::TensorInfo(1, 1, arm_compute::DataType::F32));
  bias1->allocator()->allocate();

  std::memcpy(weights0->buffer(), weights0_data, sizeof(weights0_data));
  std::memcpy(bias0->buffer(), bias0_data, sizeof(bias0_data));
  std::memcpy(weights1->buffer(), weights1_data, sizeof(weights1_data));
  std::memcpy(bias1->buffer(), bias1_data, sizeof(bias1_data));

  auto layer0 = std::make_shared<Layer>(Layer{
      .id = "layer0",
      .type = LayerType::Linear,
      .params = {},
      .hparams = {},
      .input_shape = {2},
      .output_shape = {2},
  });
  layer0->params["weight"] = std::move(weights0);
  layer0->params["bias"] = std::move(bias0);
  layer0->hparams["in_features"] = 2.0f;
  layer0->hparams["out_features"] = 2.0f;

  auto relu0 = std::make_shared<Layer>(Layer{
      .id = "relu0",
      .type = LayerType::ReLU,
      .params = {},
      .hparams = {},
      .input_shape = {2},
      .output_shape = {2},
  });

  auto layer1 = std::make_shared<Layer>(Layer{
      .id = "layer1",
      .type = LayerType::Linear,
      .params = {},
      .hparams = {},
      .input_shape = {2},
      .output_shape = {1},
  });
  layer1->params["weight"] = std::move(weights1);
  layer1->params["bias"] = std::move(bias1);
  layer1->hparams["in_features"] = 2.0f;
  layer1->hparams["out_features"] = 1.0f;

  auto relu1 = std::make_shared<Layer>(Layer{
      .id = "relu1",
      .type = LayerType::ReLU,
      .params = {},
      .hparams = {},
      .input_shape = {1},
      .output_shape = {1},
  });

  dag->layers["layer0"] = layer0;
  dag->layers["relu0"] = relu0;
  dag->layers["layer1"] = layer1;
  dag->layers["relu1"] = relu1;

  /* == Define execution units == */
  ExecutionUnit layer0_eu0{
      .id = "layer0::eu0",
      .layer = dag->layers["layer0"],
      .assigned_device = "device0",
      .input_requirements = {},
      .output_range = {0, 2},
      .forward_table = {
          {.dest_eu_id = "relu0::eu0", .required_range = {0, 2}},
      },
      .expected_input_shape = {2},
      .expected_output_shape = {2},
      .is_leaf = false,
      .is_root = true,
  };
  ExecutionUnit relu0_eu0{
      .id = "relu0::eu0",
      .layer = dag->layers["relu0"],
      .assigned_device = "device0",
      .input_requirements = {},
      .output_range = {0, 2},
      .forward_table = {
          {.dest_eu_id = "layer1::eu0", .required_range = {0, 2}},
      },
      .expected_input_shape = {2},
      .expected_output_shape = {2},
      .is_leaf = false,
      .is_root = false,
  };
  ExecutionUnit layer1_eu0{
      .id = "layer1::eu0",
      .layer = dag->layers["layer1"],
      .assigned_device = "device0",
      .input_requirements = {},
      .output_range = {0, 1},
      .forward_table = {
          {.dest_eu_id = "relu1::eu0", .required_range = {0, 1}},
      },
      .expected_input_shape = {2},
      .expected_output_shape = {1},
      .is_leaf = false,
      .is_root = false,
  };
  ExecutionUnit relu1_eu0{
      .id = "relu1::eu0",
      .layer = dag->layers["relu1"],
      .assigned_device = "device0",
      .input_requirements = {},
      .output_range = {0, 1},
      .forward_table = {},
      .expected_input_shape = {1},
      .expected_output_shape = {1},
      .is_leaf = true,
      .is_root = false,
  };

  dag->eus["layer0::eu0"] = layer0_eu0;
  dag->eus["relu0::eu0"] = relu0_eu0;
  dag->eus["layer1::eu0"] = layer1_eu0;
  dag->eus["relu1::eu0"] = relu1_eu0;

  return dag;
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
  env->ReleaseStringUTFChars(model_dag_path_jstr, model_dag_path_c);

  const char *device_info_c = env->GetStringUTFChars(device_info_jstr, nullptr);
  const auto device_info_str = std::string(device_info_c);
  env->ReleaseStringUTFChars(device_info_jstr, device_info_c);

  const char *devices_c = env->GetStringUTFChars(devices_jstr, nullptr);
  const auto devices_str = std::string(devices_c);
  env->ReleaseStringUTFChars(devices_jstr, devices_c);

  /* Create a ModelDAG object */
  auto dag = load_model_dag(model_dag_path_str);
  /* Create a DeviceInfo object */
  auto device_info = std::make_unique<DeviceInfo>();
  device_info->id = "device0";
  // TODO: Parse this from the device_info_str JSON string

  /* Create a device list */
  std::vector<DeviceInfo> devices_list{};
  // TODO: Parse the devices_str JSON string

  /* Initialize EdgeFlow */
  bool result = g_edgeflow.initialize(
      std::move(dag),
      std::move(device_info),
      devices_list);

  // Check if EdgeFlow was initialized successfully
  if (!result) {
    __android_log_print(ANDROID_LOG_ERROR, "initializeEdgeFlow",
                        "Failed to initialize EdgeFlow");
  }

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
                          "Invalid input data: '%s'", input_str.c_str());
      env->ReleaseStringUTFChars(input, input_c);
      return JNI_FALSE;
    }

    memcpy(input_tensor->buffer(), test_data, sizeof(test_data));
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
  jmethodID method = env->GetMethodID(cls, "onInferenceComplete", "([FLjava/lang/String;)V");
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
