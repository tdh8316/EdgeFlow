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

using DeviceID = std::string;
using LayerID = std::string;
using ExecutionUnitID = std::string;
using ParamsT = std::unordered_map<std::string, std::unique_ptr<arm_compute::Tensor>>;
using HyperParamsT = std::unordered_map<std::string, float>;

enum class LayerType : uint8_t {
  ReLU,
  // Sigmoid,
  // BatchNorm,
  // Concatenation,
  // Convolution,
  // Flatten,
  // Identity,
  Linear, // Fully Connected
  // PoolingAvg,
  // PoolingMax,
  // Reshape,
};

struct Layer;
struct Range;
struct InputRequirement;
struct ForwardTableEntry;
struct ExecutionUnit;

struct Layer {
  LayerID id;

  LayerType type;

  ParamsT params;
  HyperParamsT hparams;

  arm_compute::TensorShape input_shape, output_shape;
};

/// Required input range to compute the assigned output partition of the
/// execution unit The range is defined as [start, end).
struct Range {
  int start = 0; // inclusive
  int end = 0;   // exclusive

  /// The number of elements in the range
  constexpr int num_elements() const noexcept { return end - start; }

  /// Check if the range is valid. (start < end)
  constexpr bool valid() const noexcept { return start < end; }

  /// Check if the range overlaps with another range.
  constexpr bool overlaps(const Range &other) const noexcept {
    return (start < other.end) && (end > other.start);
  }

  constexpr bool operator==(const Range &other) const noexcept {
    return (start == other.start) && (end == other.end);
  }
};

/// Single input requirement of the execution unit
struct InputRequirement {
  // Source execution unit ID where the partial input comes from.
  ExecutionUnitID src_eu_id;

  // Required range from `src_eu_id`'s output
  Range src_range; // May go outside [0, H)
};

/// ForwardTable tells the execution units where to forward its output features.
/// Notice that the next execution unit might only require parts of the results.
/// Therefore, the forward table should include both the target and the required
/// range of its output.
struct ForwardTableEntry {
  ExecutionUnitID dest_eu_id; // The destination execution unit ID
  Range required_range;       // The required range of this execution unit's output
};

struct ExecutionUnit {
  ExecutionUnitID id;

  std::shared_ptr<Layer> layer; // Pointer to the layer this execution unit belongs to
  DeviceID assigned_device;

  std::unordered_map<std::string, InputRequirement> input_requirements;

  Range output_range;

  std::vector<ForwardTableEntry> forward_table;

  arm_compute::TensorShape expected_input_shape, expected_output_shape;

  bool is_leaf, is_root;

  /* == Optional fields for the convolution layer == */
  // Pre-padding amounts calculated by EdgeFlow based on paper's Eq. (5) & (6)
  int prepad_top = 0;    // Eq. (5) in the paper (upper padding)
  int prepad_bottom = 0; // Eq. (6) in the paper (bottom padding)
  int prepad_left = 0;
  int prepad_right = 0;

  const LayerType &get_type() const {
    return layer->type;
  }

  const arm_compute::Tensor *get_param(const std::string &name) const {
    auto it = layer->params.find(name);
    if (it != layer->params.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  float *get_hparam(const std::string &name) const {
    auto it = layer->hparams.find(name);
    if (it != layer->hparams.end()) {
      return &it->second;
    }
    return nullptr;
  }
};

struct ModelDAG {
  std::string name;

  std::unordered_map<LayerID, std::shared_ptr<Layer>> layers;
  std::unordered_map<ExecutionUnitID, ExecutionUnit> eus;

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
