#ifndef EDGEFLOW_NETWORKEVENTHANDLER_H
#define EDGEFLOW_NETWORKEVENTHANDLER_H

#include "edgeflow/ComputationEngine.h"
#include "edgeflow/DataTypes.h"
#include "edgeflow/EdgeFlow.h"
#include "edgeflow/Orchestrator.h"
#include <thread>

class Orchestrator;

class NetworkEventHandler {
public:
  NetworkEventHandler(Orchestrator &orch,
                      const DeviceInfo &device_info,
                      const DeviceMap &device_map);
  ~NetworkEventHandler();

  /// Start listening for incoming connections
  /// @param port The port to listen on
  void start_listening(unsigned int port);

  /// Stop listening for incoming connections
  void stop_listening();

  /// Send an intermediate result to another device
  /// @param dest_device_id The ID of the destination device
  /// @param dest_eu_id The ID of the destination execution unit
  /// @param data The intermediate result tensor to send
  void send_intermediate_result(const DeviceID &dest_device_id,
                                const ExecutionUnitID &dest_eu_id,
                                const arm_compute::Tensor &data);

  /// Callback function to be called when an intermediate result is received
  /// @param dest_eu_id The ID of the destination execution unit
  /// @param data The intermediate result tensor received
  void
  on_receive_intermediate_result(const ExecutionUnitID &dest_eu_id,
                                 std::unique_ptr<arm_compute::Tensor> data);

private:
  void listener_loop();
  // void handle_client_connection(Socket client_socket);

  const DeviceInfo &device_info_;
  const DeviceMap &device_map_;

  Orchestrator &orch_;

  // Socket server_socket_;
  std::thread listener_thread_;
  std::atomic<bool> stop_flag_{};
};

#endif // EDGEFLOW_NETWORKEVENTHANDLER_H
