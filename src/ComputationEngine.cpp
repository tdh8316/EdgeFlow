#include <utility>

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "edgeflow/ComputationEngine.h"

ComputationEngine::ComputationEngine(Orchestrator *orch,
                                     std::shared_ptr<ModelDAG> dag)
    : orch_(orch), dag_(std::move(dag)) {
  for (unsigned int i = 0; i < num_workers_; ++i) {
    worker_threads_.emplace_back(&ComputationEngine::worker_thread_loop, this);
  }
  __android_log_print(ANDROID_LOG_INFO, "ComputationEngine::ComputationEngine",
                      "ComputationEngine initialized with %u worker threads",
                      num_workers_);
}

ComputationEngine::~ComputationEngine() {
  stop_ = true;
  for (auto &worker: worker_threads_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ComputationEngine::submit_task(
        std::shared_ptr<ExecutionUnit> eu,
        std::unique_ptr<arm_compute::Tensor> input) {
  auto task = std::make_unique<Task>(std::move(eu), std::move(input));
  task_queue_.push(std::move(task));
}

void ComputationEngine::worker_thread_loop() {
  while (!stop_) {
    auto task = task_queue_.pop();

    // 1. Pre-process input tensor
    // TODO: Pre-process input tensor if needed

    // 2. Execute the operator for the execution unit
    std::unique_ptr<arm_compute::Tensor> output =
            execute_operator(*task->eu, *task->input);
    if (output) {
      orch_->on_computation_complete(task->eu, std::move(output));
    } else {
      __android_log_print(
              ANDROID_LOG_ERROR, "ComputationEngine::worker_thread_loop",
              "No output produced for execution unit %s", task->eu->id.c_str());
    }
  }

  __android_log_print(ANDROID_LOG_INFO, "ComputationEngine::worker_thread_loop",
                      "Worker thread stopped");
}

std::unique_ptr<arm_compute::Tensor>
ComputationEngine::execute_operator(const ExecutionUnit &eu,
                                    arm_compute::Tensor &input) {
  auto output = std::make_unique<arm_compute::Tensor>();
  output->allocator()->init(arm_compute::TensorInfo(
          eu.expected_output_shape, 1, arm_compute::DataType::F32));
  output->allocator()->allocate();

  switch (eu.op.type) {
    case OperatorType::Activation: {
      const auto &params = std::get<ActivationParams>(*eu.op.params);
      switch (params.type) {
        case ActivationType::ReLU: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(&input, output.get(),
                                     arm_compute::ActivationFunction::RELU);
          activation_layer.run();
          break;
        }
        case ActivationType::Sigmoid: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(&input, output.get(),
                                     arm_compute::ActivationFunction::LOGISTIC);
          activation_layer.run();
          break;
        }
        case ActivationType::Softmax: {
          arm_compute::NESoftmaxLayer softmax_layer;
          softmax_layer.configure(&input, output.get());
          softmax_layer.run();
          break;
        }
        case ActivationType::Swish:
        case ActivationType::SiLU: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(&input, output.get(),
                                     arm_compute::ActivationFunction::SWISH);
          activation_layer.run();
          break;
        }
          // TODO: Add other activation functions
        default: {
          __android_log_print(
                  ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
                  "Unsupported activation type for execution unit %s", eu.id.c_str());
          return nullptr;
        }
      }
      break;
    }
    case OperatorType::BatchNorm: {
      const auto &params = std::get<BatchNormParams>(*eu.op.params);
      arm_compute::Tensor var;
      // TODO: Initialize var tensor
      arm_compute::NEBatchNormalizationLayer batch_norm_layer;
      batch_norm_layer.configure(&input, output.get(), params.mean.get(),
                                 &var, params.beta.get(),
                                 params.gamma.get());
      batch_norm_layer.run();
      break;
    }
    case OperatorType::Concatenation: {
      const auto &params = std::get<ConcatenationParams>(*eu.op.params);
      arm_compute::NEConcatenateLayer concat_layer;
      // TODO: Input must be a vector of tensors, which is incompatible with the current implementation
      __android_log_print(
              ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
              "Concatenation operation not implemented for execution unit %s", eu.id.c_str());
      return nullptr;
      break;
    }
    case OperatorType::Convolution: {
      const auto &params = std::get<ConvolutionParams>(*eu.op.params);
      arm_compute::NEConvolutionLayer conv_layer;
      auto conv_info = arm_compute::PadStrideInfo(
              params.stride_w, params.stride_h,
              params.padding_w, params.padding_h,
              arm_compute::DimensionRoundingType::CEIL);
      conv_layer.configure(
              &input, params.weight.get(), params.bias.get(), output.get(), conv_info);
      conv_layer.run();
      break;
    }
    case OperatorType::Flatten: {
      // TODO: Implement flatten operation
      __android_log_print(
              ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
              "Flatten operation not implemented for execution unit %s", eu.id.c_str());
      return nullptr;
      break;
    }
    case OperatorType::Identity: {
      output->copy_from(input);
    }
    case OperatorType::Linear: {
      const auto &params = std::get<LinearParams>(*eu.op.params);
      arm_compute::NEFullyConnectedLayer fc_layer;
      // TODO: Maybe fused activation?
      fc_layer.configure(&input, params.weight.get(), params.bias.get(),
                         output.get());
      fc_layer.run();
      break;
    }
    case OperatorType::PoolingAvg: {
      const auto &params = std::get<PoolingParams>(*eu.op.params);
      arm_compute::NEPoolingLayer pooling_layer;
      auto pool_info = arm_compute::PoolingLayerInfo(); // TODO: Set pooling info
      pooling_layer.configure(&input, output.get(), pool_info);
      pooling_layer.run();
      break;
    }
    case OperatorType::PoolingMax: {
      const auto &params = std::get<PoolingParams>(*eu.op.params);
      arm_compute::NEPoolingLayer pooling_layer;
      auto pool_info = arm_compute::PoolingLayerInfo(); // TODO: Set pooling info
      pooling_layer.configure(&input, output.get(), pool_info);
      pooling_layer.run();
      break;
    }
    case OperatorType::Reshape: {
      // TODO: Implement reshape operation
      __android_log_print(
              ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
              "Reshape operation not implemented for execution unit %s", eu.id.c_str());
      return nullptr;
      break;
    }
      // TODO: Add other operator types
    default: {
      __android_log_print(
              ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
              "Unsupported operator type for execution unit %s", eu.id.c_str());
      return nullptr;
    }
  }

  __android_log_print(ANDROID_LOG_INFO, "ComputationEngine::execute_operator",
                      "Operator executed successfully for execution unit %s for layer %s",
                      eu.id.c_str(), eu.layer_id.c_str());
  print_tensor(*output, "ComputationEngine::execute_operator::output");
  return output;
}
