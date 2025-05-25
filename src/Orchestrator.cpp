#include "edgeflow/Orchestrator.h"

Orchestrator::Orchestrator(std::shared_ptr<ModelDAG> dag,
                           std::shared_ptr<DeviceInfo> device_info,
                           std::shared_ptr<DeviceMap> device_map)
    : dag_(std::move(dag)), device_info_(std::move(device_info)),
      device_map_(std::move(device_map)),
      computation_engine_(new ComputationEngine(this, dag_)),
      network_event_handler_(
              new NetworkEventHandler(this, device_info_, device_map_)) {
  // Initialize the network listener
  network_event_handler_->start_listening(device_info_->port);

  // Initialize the input states for each execution unit
  for (const auto &eu_map: dag_->eus) {
    const ExecutionUnit &eu = eu_map.second;
    if (eu.device_id == device_info_->id) {
      // Initialize the input state for the execution unit
      input_states_[eu.id].num_expected = eu.input_requirements.size();
    }
  }
}

Orchestrator::~Orchestrator() = default;

void Orchestrator::register_inference_complete_callback(
        Orchestrator::Callback inference_complete_callback) {
  inference_complete_callback_ = std::move(inference_complete_callback);
}

bool Orchestrator::start_inference(std::unique_ptr<arm_compute::Tensor> input) {
  std::lock_guard<std::mutex> lock(orch_mtx_);

  // Clean up the previous outputs
  collected_final_outputs_.clear();
  int exit_eu_on_this_device = 0;
  for (std::pair<const ExecutionUnitID, InputState> &eu_state: input_states_) {
    const auto &eu_id = eu_state.first;
    auto eu = get_execution_unit(eu_id);
    if (!eu) {
      __android_log_print(ANDROID_LOG_ERROR, "Orchestrator::start_inference",
                          "Execution unit %s not found", eu_id.c_str());
      return false;
    }

    // Clean up the input state
    auto &input_state = eu_state.second;
    input_state.received.clear();
    input_state.num_received = 0;

    // Find the number of leaf execution units on this device
    if (eu->is_leaf && eu->device_id == device_info_->id) {
      ++exit_eu_on_this_device;
    }

    // Handle the root execution unit (input layer)
    if (eu->is_root && eu->device_id == device_info_->id) {
      if (eu->input_requirements.empty()) {
        // Start the inference on the root execution unit
        computation_engine_->submit_task(eu, std::move(input));
      } else {
        __android_log_print(
                ANDROID_LOG_ERROR, "Orchestrator::start_inference",
                "Input requirements for execution unit %s not empty, "
                "which should be empty for the root execution unit",
                eu_id.c_str());
        return false;
      }
    }
  }
  num_pending_leaf_eus_.store(exit_eu_on_this_device);
  if (num_pending_leaf_eus_ == 0) {
    __android_log_print(ANDROID_LOG_WARN, "Orchestrator::start_inference", "No leaf execution units on this device!");
  }

  return true;
}

void Orchestrator::on_receive_intermediate_result(
        std::unique_ptr<ExecutionUnitID> src_eu_id,
        std::unique_ptr<ExecutionUnitID> dest_eu_id,
        std::unique_ptr<arm_compute::Tensor> data) {
  // TODO
}

void Orchestrator::on_computation_complete(
        const std::shared_ptr<ExecutionUnit> &completed_eu,
        std::unique_ptr<arm_compute::Tensor> output) {
  dispatch_output(*completed_eu, output);

  // Check if the output is from a leaf execution unit
  if (completed_eu->is_leaf) {
    std::lock_guard<std::mutex> lock(collected_final_outputs_mtx_);

    // Store the output tensor for the leaf execution unit
    collected_final_outputs_[completed_eu->id] = std::move(output);

    int remaining = num_pending_leaf_eus_.fetch_sub(1) - 1;
    __android_log_print(
            ANDROID_LOG_INFO, "Orchestrator::on_computation_complete",
            "A leaf execution unit is completed; remaining leaf execution units: %d", remaining);

    if (remaining == 0) {
      if (inference_complete_callback_) {
        __android_log_print(
                ANDROID_LOG_INFO, "Orchestrator::on_computation_complete",
                "All leaf execution units completed; invoke inference_complete_callback_");
        // Invoke the callback with the collected final outputs
        // TODO: Combine the outputs before invoking the callback
        for (const auto &output_pair: collected_final_outputs_) {
          const auto &eu_id = output_pair.first;
          const auto &output_tensor = output_pair.second;
          inference_complete_callback_(*output_tensor);
        }
      } else {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Orchestrator::on_computation_complete",
                            "Inference completed, but no callback registered");
      }
    }
  }
}

void Orchestrator::check_and_run_eu(const ExecutionUnitID &eu_id) {}

std::unique_ptr<arm_compute::Tensor>
Orchestrator::assemble_input_for_eu(const ExecutionUnit &eu,
                                    InputState &input_state) {
  return {};
}

void Orchestrator::dispatch_output(
        const ExecutionUnit &src_eu,
        const std::unique_ptr<arm_compute::Tensor> &output) {
  const auto &forward_table = src_eu.forward_table.entries;
  if (forward_table.empty() && !(src_eu.is_leaf)) {
    __android_log_print(ANDROID_LOG_ERROR, "Orchestrator::dispatch_output",
                        "No forward table entries for non-leaf execution unit %s",
                        src_eu.id.c_str());
  }

  for (const auto &entry: forward_table) {
    const auto &dest_eu_id = entry.dest_eu_id;
    const auto &required_range = entry.required_range;

    // Find the destination execution unit
    auto dest_eu = get_execution_unit(dest_eu_id);
    if (!dest_eu) {
      __android_log_print(ANDROID_LOG_ERROR, "Orchestrator::dispatch_output",
                          "Invalid destination execution unit %s for source %s",
                          dest_eu_id.c_str(), src_eu.id.c_str());
      continue;
    }

    // TODO: Check if the output range matches the required range
    // Currently, dispatch the entire output tensor
    // In the future, we may need to slice the output tensor according to the required range
    {
      auto forwarding_tensor = std::make_unique<arm_compute::Tensor>();
      forwarding_tensor->allocator()->init(arm_compute::TensorInfo(
              dest_eu->expected_input_shape, 1, arm_compute::DataType::F32));
      forwarding_tensor->allocator()->allocate();
      forwarding_tensor->copy_from(*output);
      // print_tensor(*output, "Output tensor to be dispatched");
      __android_log_print(ANDROID_LOG_INFO, "Orchestrator::dispatch_output",
                          "Dispatching output tensor from %s -> %s",
                          src_eu.id.c_str(), dest_eu_id.c_str());
      print_tensor(*forwarding_tensor, "Orchestrator::dispatch_output::forwarding_tensor");

      if (dest_eu->device_id == device_info_->id) {
        // If the destination execution unit is on the same device,
        // we can directly submit the task to the computation engine
        computation_engine_->submit_task(dest_eu, std::move(forwarding_tensor));
      } else {
        // If the destination execution unit is on a different device,
        // we need to send the intermediate result over the network
        network_event_handler_->send_intermediate_result(
                dest_eu->device_id, dest_eu_id, *forwarding_tensor);
      }
    }
  }
}

std::shared_ptr<ExecutionUnit>
Orchestrator::get_execution_unit(const ExecutionUnitID &eu_id) const {
  auto it = dag_->eus.find(eu_id);
  if (it != dag_->eus.end()) {
    return std::make_shared<ExecutionUnit>(it->second);
  }
  __android_log_print(ANDROID_LOG_ERROR, "Orchestrator::get_execution_unit",
                      "Execution unit %s not found", eu_id.c_str());
  return nullptr;
}
