#ifndef EDGEFLOW_DATATYPES_H
#define EDGEFLOW_DATATYPES_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include <android/log.h>
#include <string>
#include <unordered_map>
#include <vector>

/// Print tensor for debugging
void print_tensor(const arm_compute::Tensor &tensor,
                  const std::string &name = "tensor");

/* Primitive types */
using DeviceID = std::string;
using LayerID = std::string;
using ExecutionUnitID = std::string;

/// Activation function type
enum class ActivationType : unsigned int {
  ReLU,
  Sigmoid,
  Softmax,
  Swish,
  SiLU,
};

/// Operation type of the current layer
enum class OperatorType : unsigned int {
  Activation,
  BatchNorm,
  Concatenation,
  Convolution,
  Flatten,
  Identity,
  Linear, // Fully Connected
  PoolingAvg,
  PoolingMax,
  Reshape,
};

/// Required input range to compute the assigned output partition of the
/// execution unit The range is defined as [start, end)
struct Range {
  int start = 0; // inclusive
  int end = 0;   // exclusive

  Range() = default;

  Range(int start, int end) : start(start), end(end) {}

  /// The number of elements in the range
  int num_elements() const noexcept { return end - start; }

  /// Check if the range is valid (start < end)
  bool valid() const noexcept { return start < end; }

  /// Check if the range overlaps with another range
  bool overlaps(const Range &other) const noexcept {
    return (start < other.end) && (end > other.start);
  }

  bool operator==(const Range &other) const noexcept {
    return (start == other.start) && (end == other.end);
  }

  bool operator!=(const Range &other) const noexcept {
    return !(*this == other);
  }
};

/// Activation parameters
struct ActivationParams {
  ActivationType type;
};

/// Linear parameters
struct LinearParams {
  unsigned int in_features = 0, out_features = 0;

  std::unique_ptr<arm_compute::Tensor> weight, bias;
};

/// Convolution parameters
struct ConvolutionParams {
  unsigned int kernel_h = 0, kernel_w = 0;
  unsigned int stride_h = 0, stride_w = 0;
  unsigned int padding_h = 0, padding_w = 0; // Original padding

  // Pre-padding amounts calculated by EdgeFlow based on paper's Eq. (5) & (6)
  // These are applied before the core convolution operation if the layer's
  // own padding is set to 0.
  int prepad_top = 0;    // Eq. (5) in the paper (upper padding)
  int prepad_bottom = 0; // Eq. (6) in the paper (bottom padding)
  int prepad_left = 0;   // Assuming partitioning is only along height,
  int prepad_right = 0;  // these might always be original padding_w or 0.

  std::unique_ptr<arm_compute::Tensor> weight, bias;
};

struct PoolingParams {
  unsigned int pool_h = 0, pool_w = 0;
  unsigned int stride_h = 0, stride_w = 0;
  unsigned int pad_h = 0, pad_w = 0; // Original padding

  // Pre-padding amounts calculated by EdgeFlow based on paper's Eq. (5) & (6)
  int prepad_top = 0;    // Eq. (5) in the paper (upper padding)
  int prepad_bottom = 0; // Eq. (6) in the paper (bottom padding)
  int prepad_left = 0;
  int prepad_right = 0;
};

struct BatchNormParams {
  std::unique_ptr<arm_compute::Tensor> mean, variance, beta, gamma;
};

struct ConcatenationParams {
  unsigned int axis = 0; // Axis along which to concatenate
};

// TODO: Operation-specific parameters

using OperatorParams =
        std::variant<ActivationParams, LinearParams, ConvolutionParams,
                     PoolingParams, BatchNormParams, ConcatenationParams
                     // TODO: Add other operation-specific parameters
                     >;

/// Computation operator including operation type of the current layer
/// as well as the parameters for the layer
struct Operator {
  // Operation type of the current execution unit of the layer
  OperatorType type;

  // Operation-specific hyper-parameters
  std::shared_ptr<OperatorParams> params;
};

/// Single input requirement of the execution unit
struct InputRequirement {
  // Source execution unit ID where the input comes from
  ExecutionUnitID src_eu_id;

  // Required range from `src_eu_id`'s output
  // used to compute the assigned output partition of the execution unit
  Range src_range; // May go outside [0, H)
};

struct ExecutionUnit;

struct ForwardTableEntry {
  ExecutionUnitID dest_eu_id; // The destination execution unit ID
  Range required_range;       // The required range of this execution unit's output
};

/// ForwardTable tells the execution units where to forward its output features.
/// Notice that the next execution unit might only require parts of the results.
/// Therefore, the forward table should include both the target and the required
/// range of its output.
struct ForwardTable {
  // The next execution unit and range of the output it requires
  std::vector<ForwardTableEntry> entries;
};

/// Defines the execution unit
/// corresponding to generate a part of the output of the layer
struct ExecutionUnit {
  // The original layer ID which the execution unit belongs to
  LayerID layer_id;

  // The device ID on which the execution unit is assigned
  DeviceID device_id;

  // ID of the execution unit
  ExecutionUnitID id;

  // The required input might be from multiple execution units
  std::vector<InputRequirement> input_requirements;

  // The output range this unit is responsible for
  // w.r.t. its original layer's output
  Range output_range;

  // Operation of the current execution unit of the layer
  Operator op;

  // Forwarding table of the current execution unit
  ForwardTable forward_table;

  // For Orchestrator, assembled input information
  arm_compute::TensorShape expected_input_shape;

  // For Orchestrator, information of output this unit will produce
  arm_compute::TensorShape expected_output_shape;

  // Indicate if the execution unit is leaf or root
  bool is_leaf, is_root;

  /// Find an input requirement by its source execution unit ID
  const InputRequirement *
  find_input_requirement_from_src(const ExecutionUnitID &src_eu_id) {
    for (const auto &input_req: input_requirements) {
      if (input_req.src_eu_id == src_eu_id) {
        return &input_req;
      }
    }
    return nullptr;
  }
};

/// Defines the logical layer
/// corresponding to generate a part of the output of the model
/// The layer consists of multiple execution units
struct Layer {
  // The layer ID
  LayerID id;

  // Original operation type of the layer
  OperatorType type;

  // Original operation-specific hyper-parameters
  std::shared_ptr<OperatorParams> params;

  // Layer input and output shapes
  arm_compute::TensorShape input_shape, output_shape;

  // The (children) execution units of the layer
  // std::vector<ExecutionUnit> execution_units;
};

/// Defines the model as a DAG
struct ModelDAG {
  // The model name
  std::string name;

  // Layer mapping
  std::unordered_map<LayerID, Layer> layers;

  // Execution unit mapping
  std::unordered_map<ExecutionUnitID, ExecutionUnit> eus;

  // Adjacency list of the layer-wise model
  std::unordered_map<LayerID, std::vector<LayerID>> layer_wise_graph;

  // Model input and output shapes
  arm_compute::TensorShape input_shape, output_shape;
};

/// Device information
struct DeviceInfo {
  DeviceID id;
  std::string ip_address;
  unsigned int port;
};

using DeviceMap = std::unordered_map<DeviceID, DeviceInfo>;

#endif // EDGEFLOW_DATATYPES_H
