#include "edgeflow/NetworkEventHandler.h"

NetworkEventHandler::NetworkEventHandler(
    Orchestrator &orch,
    const DeviceInfo &device_info,
    const DeviceMap &device_map)
    : orch_(orch), device_info_(device_info),
      device_map_(device_map) {
  // TODO
}

NetworkEventHandler::~NetworkEventHandler() {
  stop_listening();
  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }
  __android_log_print(ANDROID_LOG_INFO, "NetworkEventHandler::~NetworkEventHandler",
                      "NetworkEventHandler destroyed");
}

void NetworkEventHandler::start_listening(unsigned int port) {}

void NetworkEventHandler::stop_listening() {}

void NetworkEventHandler::send_intermediate_result(
    const DeviceID &dest_device_id,
    const ExecutionUnit &dest_eu,
    std::unique_ptr<arm_compute::Tensor> input) {}

void NetworkEventHandler::on_receive_intermediate_result(
    const ExecutionUnitID &dest_eu_id,
    std::unique_ptr<arm_compute::Tensor> data) {}

void NetworkEventHandler::listener_loop() {}
