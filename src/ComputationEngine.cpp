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

ComputationEngine::ComputationEngine(Orchestrator &orch,
                                     const ModelDAG &dag)
    : orch_(orch),
      dag_(dag),
      num_workers_(std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency() * 0.75))) {
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
    const ExecutionUnit &eu,
    std::unique_ptr<arm_compute::Tensor> input) {
  auto task = std::make_unique<Task>(eu, std::move(input));
  task_queue_.push(std::move(task));
}

void ComputationEngine::worker_thread_loop() {
  while (!stop_) {
    const auto task = task_queue_.pop();

    // 1. Pre-process input tensor
    // TODO: Pre-process input tensor if needed

    // 2. Execute the operator for the execution unit
    auto output = execute_operator(task->eu, std::move(task->input));
    if (output) {
      orch_.on_computation_complete(task->eu, std::move(output));
    } else {
      __android_log_print(
          ANDROID_LOG_ERROR, "ComputationEngine::worker_thread_loop",
          "No output produced for execution unit %.*s",
          static_cast<int>(task->eu.id.size()), task->eu.id.data());
    }
  }

  __android_log_print(ANDROID_LOG_INFO, "ComputationEngine::worker_thread_loop",
                      "Worker thread stopped");
}

std::unique_ptr<arm_compute::Tensor>
ComputationEngine::execute_operator(const ExecutionUnit &eu,
                                    std::unique_ptr<arm_compute::Tensor> input) {
  auto output = std::make_unique<arm_compute::Tensor>();
  output->allocator()->init(arm_compute::TensorInfo(
      eu.expected_output_shape, 1, arm_compute::DataType::F32));
  output->allocator()->allocate();

  switch (eu.get_type()) {
    case LayerType::ReLU: {
      arm_compute::NEActivationLayer activation_layer;
      activation_layer.configure(
          input.get(),
          output.get(),
          arm_compute::ActivationFunction::RELU);
      activation_layer.run();
      break;
    }
    case LayerType::Linear: {
      arm_compute::NEFullyConnectedLayer fc_layer;
      fc_layer.configure(
          input.get(),
          eu.get_param("weight"),
          eu.get_param("bias"),
          output.get());
      fc_layer.run();
      break;
    }
    default: {
      __android_log_print(
          ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
          "Unsupported operator type for execution unit %.*s",
          static_cast<int>(eu.id.size()), eu.id.data());
      return nullptr;
    }
  }

  return output;
}

/*
std::unique_ptr<arm_compute::Tensor>
_execute_operator(const ExecutionUnit &eu,
                  std::unique_ptr<arm_compute::Tensor> input) {
  auto output = std::make_unique<arm_compute::Tensor>();
  output->allocator()->init(arm_compute::TensorInfo(
      eu.expected_output_shape, 1, arm_compute::DataType::F32));
  output->allocator()->allocate();

  switch (eu.op.type) {
    case OperatorType::Activation: {
      const auto &params = std::get<ActivationHParams>(*eu.op.hparams);
      switch (params.type) {
        case ActivationType::ReLU: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(input.get(), output.get(),
                                     arm_compute::ActivationFunction::RELU);
          activation_layer.run();
          break;
        }
        case ActivationType::Sigmoid: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(input.get(), output.get(),
                                     arm_compute::ActivationFunction::LOGISTIC);
          activation_layer.run();
          break;
        }
        case ActivationType::Softmax: {
          arm_compute::NESoftmaxLayer softmax_layer;
          softmax_layer.configure(input.get(), output.get());
          softmax_layer.run();
          break;
        }
        case ActivationType::Swish:
        case ActivationType::SiLU: {
          arm_compute::NEActivationLayer activation_layer;
          activation_layer.configure(input.get(), output.get(),
                                     arm_compute::ActivationFunction::SWISH);
          activation_layer.run();
          break;
        }
          // TODO: Add other activation functions here
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
      const auto &params = std::get<BatchNormHParams>(*eu.op.hparams);
      arm_compute::Tensor var;
      // TODO: Initialize var tensor
      arm_compute::NEBatchNormalizationLayer batch_norm_layer;
      batch_norm_layer.configure(input.get(), output.get(), params.mean.get(),
                                 &var, params.beta.get(),
                                 params.gamma.get());
      batch_norm_layer.run();
      break;
    }
    case OperatorType::Concatenation: {
      const auto &params = std::get<ConcatenationHParams>(*eu.op.hparams);
      arm_compute::NEConcatenateLayer concat_layer;
      // TODO: Input must be a vector of tensors, which is incompatible with the current implementation
      __android_log_print(
          ANDROID_LOG_ERROR, "ComputationEngine::execute_operator",
          "Concatenation operation not implemented for execution unit %s", eu.id.c_str());
      return nullptr;
      break;
    }
    case OperatorType::Convolution: {
      const auto &params = std::get<ConvolutionHParams>(*eu.op.hparams);
      arm_compute::NEConvolutionLayer conv_layer;
      auto conv_info = arm_compute::PadStrideInfo(
          params.stride_w, params.stride_h,
          params.padding_w, params.padding_h,
          arm_compute::DimensionRoundingType::CEIL);
      conv_layer.configure(
          input.get(), params.weight.get(), params.bias.get(), output.get(), conv_info);
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
      output->copy_from(*input);
    }
    case OperatorType::Linear: {
      const auto &params = std::get<LinearHParams>(*eu.op.hparams);
      arm_compute::NEFullyConnectedLayer fc_layer;
      // TODO: Maybe fused activation?
      fc_layer.configure(input.get(), params.weight.get(), params.bias.get(),
                         output.get());
      fc_layer.run();
      break;
    }
    case OperatorType::PoolingAvg: {
      const auto &params = std::get<PoolingHParams>(*eu.op.hparams);
      arm_compute::NEPoolingLayer pooling_layer;
      auto pool_info = arm_compute::PoolingLayerInfo(); // TODO: Set pooling info
      pooling_layer.configure(input.get(), output.get(), pool_info);
      pooling_layer.run();
      break;
    }
    case OperatorType::PoolingMax: {
      const auto &params = std::get<PoolingHParams>(*eu.op.hparams);
      arm_compute::NEPoolingLayer pooling_layer;
      auto pool_info = arm_compute::PoolingLayerInfo(); // TODO: Set pooling info
      pooling_layer.configure(input.get(), output.get(), pool_info);
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
                      "UnitOperator executed successfully for execution unit %s for layer %s",
                      eu.id.c_str(), eu.layer_id.c_str());
  print_tensor(*output, "ComputationEngine::execute_operator::output");
  return output;
}
*/
