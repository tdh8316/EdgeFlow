#include "edgeflow/NetworkEventHandler.h"

NetworkEventHandler::NetworkEventHandler(
        Orchestrator *orch, std::shared_ptr<DeviceInfo> device_info,
        std::shared_ptr<DeviceMap> device_map)
    : orch_(orch), device_info_(std::move(device_info)),
      device_map_(std::move(device_map)) {
  // TODO
}

NetworkEventHandler::~NetworkEventHandler() {}

void NetworkEventHandler::start_listening(unsigned int port) {}

void NetworkEventHandler::stop_listening() {}

void NetworkEventHandler::send_intermediate_result(
        const DeviceID &dest_device_id, const ExecutionUnitID &dest_eu_id,
        const arm_compute::Tensor &data) {}

void NetworkEventHandler::on_receive_intermediate_result(
        std::unique_ptr<ExecutionUnitID> dest_eu_id,
        std::unique_ptr<arm_compute::Tensor> data) {}

void NetworkEventHandler::listener_loop() {}
