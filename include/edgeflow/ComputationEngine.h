#ifndef EDGEFLOW_COMPUTATIONENGINE_H
#define EDGEFLOW_COMPUTATIONENGINE_H

#include "ThreadSafeQueue.hpp"
#include "edgeflow/DataTypes.h"
#include "edgeflow/EdgeFlow.h"
#include "edgeflow/NetworkEventHandler.h"
#include "edgeflow/Orchestrator.h"
#include <thread>
#include <utility>

class Orchestrator;
class ComputationEngine {
public:
  ComputationEngine(Orchestrator *orch, std::shared_ptr<ModelDAG> dag);
  ~ComputationEngine();

  /// Computation task worker processes
  struct Task {
    std::shared_ptr<ExecutionUnit> eu;
    std::unique_ptr<arm_compute::Tensor> input;

    Task(std::shared_ptr<ExecutionUnit> eu,
         std::unique_ptr<arm_compute::Tensor> input)
        : eu(eu), input(std::move(input)) {}
  };

  /// Enqueue an execution unit for processing
  /// @param eu Execution unit to run
  /// @param input The input tensor for the execution unit
  void submit_task(std::shared_ptr<ExecutionUnit> eu,
                   std::unique_ptr<arm_compute::Tensor> input);

private:
  /// Worker thread loop.
  /// Pops tasks from the queue and executes them.
  /// After finishing the task, it calls the Orchestrator to
  /// forward the output.
  void worker_thread_loop();

  std::unique_ptr<arm_compute::Tensor>
  execute_operator(const ExecutionUnit &eu, arm_compute::Tensor &input);

  Orchestrator *orch_ = nullptr;
  std::shared_ptr<ModelDAG> dag_ = nullptr;

  ThreadSafeQueue<Task> task_queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> stop_{false};
  const unsigned int num_workers_ = std::thread::hardware_concurrency();
};

#endif // EDGEFLOW_COMPUTATIONENGINE_H
