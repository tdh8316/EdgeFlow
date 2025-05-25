#ifndef EDGEFLOW_ORCHESTRATOR_H
#define EDGEFLOW_ORCHESTRATOR_H

#include "edgeflow/ComputationEngine.h"
#include "edgeflow/DataTypes.h"
#include "edgeflow/EdgeFlow.h"
#include "edgeflow/NetworkEventHandler.h"
#include <set>

class ComputationEngine;
class NetworkEventHandler;

class Orchestrator {
public:
  Orchestrator(std::shared_ptr<ModelDAG> dag,
               std::shared_ptr<DeviceInfo> device_info,
               std::shared_ptr<DeviceMap> device_map);

  ~Orchestrator();

  /// Input state of the execution unit
  struct InputState {
    // Received intermediate results from the source execution units
    std::unordered_map<ExecutionUnitID, std::unique_ptr<arm_compute::Tensor>>
            received{};

    unsigned int num_expected = 0;
    unsigned int num_received = 0;

    std::mutex mtx{};
  };

  using Callback = std::function<void(const arm_compute::Tensor &)>;
  /// Register the callback function to be called when the inference is
  /// complete i.e., this.inference_complete_callback_ is invoked.
  void
  register_inference_complete_callback(Callback inference_complete_callback);

  /// Start the inference process
  /// @param input The input tensor to be used for inference
  /// @return true if the inference process is started successfully,
  bool start_inference(std::unique_ptr<arm_compute::Tensor> input);

  /// Callback function to be called when
  /// receives an intermediate result from another device or a local device
  /// @param dest_eu The ID of destination execution unit
  /// @param data The intermediate result tensor used as an input
  /// for the dest_eu
  void
  on_receive_intermediate_result(std::unique_ptr<ExecutionUnitID> src_eu_id,
                                 std::unique_ptr<ExecutionUnitID> dest_eu_id,
                                 std::unique_ptr<arm_compute::Tensor> data);

  /// Callback function to be called when the ComputationEngine is finished
  /// the given execution unit. The resulting tensor will be forwarded to the
  /// next execution unit.
  /// @param completed_eu The completed execution unit
  /// @param output The output tensor of the execution unit itself
  void on_computation_complete(const std::shared_ptr<ExecutionUnit> &completed_eu,
                               std::unique_ptr<arm_compute::Tensor> output);

private:
  void check_and_run_eu(const ExecutionUnitID &eu_id);

  std::unique_ptr<arm_compute::Tensor>
  assemble_input_for_eu(const ExecutionUnit &eu, InputState &input_state);

  /// Dispatch the output tensor to the next execution unit
  /// @param src_eu The execution unit that produced the output
  /// @param output The output tensor to be dispatched
  void dispatch_output(const ExecutionUnit &src_eu,
                       const std::unique_ptr<arm_compute::Tensor> &output);

  std::shared_ptr<ExecutionUnit>
  get_execution_unit(const ExecutionUnitID &eu_id) const;

  std::shared_ptr<ModelDAG> dag_ = nullptr;
  std::shared_ptr<DeviceInfo> device_info_ = nullptr;
  std::shared_ptr<DeviceMap> device_map_ = nullptr;

  ComputationEngine *computation_engine_ = nullptr;
  NetworkEventHandler *network_event_handler_ = nullptr;

  // EdgeFlow::on_inference_complete() will be assigned to this
  Callback inference_complete_callback_ = nullptr;

  // The set of execution units that are responsible for this device
  std::unordered_map<ExecutionUnitID, InputState> input_states_{};
  std::mutex orch_mtx_{};

  // Leaf execution units
  std::unordered_map<ExecutionUnitID, std::unique_ptr<arm_compute::Tensor>>
          collected_final_outputs_{};
  std::mutex collected_final_outputs_mtx_{};

  std::atomic<int> num_pending_leaf_eus_{0};
};

#endif // EDGEFLOW_ORCHESTRATOR_H
