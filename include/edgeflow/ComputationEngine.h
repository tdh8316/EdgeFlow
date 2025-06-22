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
  ComputationEngine(Orchestrator &orch, const ModelDAG &dag);
  ~ComputationEngine();

  /// Computation task worker processes
  struct Task {
    const ExecutionUnit &eu;
    std::unique_ptr<arm_compute::Tensor> input;

    Task(const ExecutionUnit &eu,
         std::unique_ptr<arm_compute::Tensor> input)
        : eu(eu), input(std::move(input)) {}
  };

  /// Enqueue an execution unit for processing to the task queue.
  /// @param eu Execution unit to run
  /// @param input The input tensor for the execution unit
  void submit_task(const ExecutionUnit &eu,
                   std::unique_ptr<arm_compute::Tensor> input);

private:
  /// Worker thread loop.
  /// Pops tasks from the queue and executes them.
  /// After finishing the task, it calls the Orchestrator to
  /// forward the output.
  void worker_thread_loop();

  /// Execute the operator for the given execution unit.
  /// This function is invoked by the `worker_thread_loop`.
  static std::unique_ptr<arm_compute::Tensor>
  execute_operator(
      const ExecutionUnit &eu,
      std::unique_ptr<arm_compute::Tensor> input);

  Orchestrator &orch_; // For calling `on_computation_complete`
  const ModelDAG &dag_;

  ThreadSafeQueue<Task> task_queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> stop_{false};
  const unsigned int num_workers_;
};

#endif // EDGEFLOW_COMPUTATIONENGINE_H
