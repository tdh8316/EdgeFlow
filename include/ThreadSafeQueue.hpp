#ifndef EDGEFLOW_THREADSAFEQUEUE_HPP
#define EDGEFLOW_THREADSAFEQUEUE_HPP

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>

template<typename T>
class ThreadSafeQueue {
public:
  ThreadSafeQueue() = default;
  ~ThreadSafeQueue() = default;

  // Non-copyable and non-movable
  ThreadSafeQueue(const ThreadSafeQueue &) = delete;
  ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;
  ThreadSafeQueue(ThreadSafeQueue &&) = delete;
  ThreadSafeQueue &operator=(ThreadSafeQueue &&) = delete;

  void push(std::unique_ptr<T> item) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      q_.push(std::move(item));
    }
    cv_.notify_one();
  }

  /// Blocking pop
  std::unique_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return !q_.empty(); });

    std::unique_ptr<T> item = std::move(q_.front());
    q_.pop();
    return item;
  }

  /// Non-blocking pop
  std::unique_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (q_.empty())
      return nullptr;

    std::unique_ptr<T> item = std::move(q_.front());
    q_.pop();
    return item;
  }

  /// Check if the queue is empty
  bool empty() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.empty();
  }

  /// Get the size of the queue
  size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.size();
  }

private:
  std::queue<std::unique_ptr<T>> q_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
};

#endif // EDGEFLOW_THREADSAFEQUEUE_HPP
