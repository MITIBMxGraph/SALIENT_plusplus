#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <future>
struct timer {
  double total_time;
  double last_time;
  bool on;
  std::string name;
  struct timezone tzp;

  timer(std::string name = "PBBS time", bool _start = true)
  : total_time(0.0), on(false), name(name), tzp({0,0}) {
    if (_start) start();
  }

  double get_time() {
    timeval now;
    gettimeofday(&now, &tzp);
    return ((double) now.tv_sec) + ((double) now.tv_usec)/1000000.;
  }

  void start () {
    on = 1;
    last_time = get_time();
  }

  double stop () {
    on = 0;
    double d = (get_time()-last_time);
    total_time += d;
    return d;
  }

  void reset() {
     total_time=0.0;
     on=0;
  }

  double get_total() {
    if (on) return total_time + get_time() - last_time;
    else return total_time;
  }

  double get_next() {
    if (!on) return 0.0;
    double t = get_time();
    double td = t - last_time;
    total_time += td;
    last_time = t;
    return td;
  }

  void report(double time, std::string str) {
    std::ios::fmtflags cout_settings = std::cout.flags();
    std::cout.precision(4);
    std::cout << std::fixed;
    std::cout << name << ": ";
    if (str.length() > 0)
      std::cout << str << ": ";
    std::cout << time << std::endl;
    std::cout.flags(cout_settings);
  }

  void total() {
    report(get_total(),"total");
    total_time = 0.0;
  }

  void reportTotal(std::string str) {
    report(get_total(), str);
  }

  void next(std::string str) {
    if (on) report(get_next(), str);
  }
};

static timer _tm;
static timer _tm101;
static timer _tm102;
static timer _tm103;
static timer _tm104;
static timer _tm105;
static timer _tm106;
static timer _tm107;
static timer _tm108;
#define startTime() //_tm.start();
#define nextTime(_string) //_tm.next(_string);










#include <pybind11/chrono.h>
#include <semaphore.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

// torch pybind?
// #include "pytorch_pybind.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>

//#include "nvToolsExt.h"

#include "range_partition_book.hpp"


using namespace torch::indexing;


struct MySemaphore {
  MySemaphore(unsigned int value) { sem_init(&sem, 0, value); }

  ~MySemaphore() noexcept(false) {
    if (sem_destroy(&sem) == -1) {
      throw std::system_error(errno, std::generic_category());
    }
  }

  void acquire() {
    if (sem_wait(&sem) == -1) {
      throw std::system_error(errno, std::generic_category());
    }
  }

  void release(std::ptrdiff_t update = 1) {
    while (update > 0) {
      if (sem_post(&sem) == -1) {
        throw std::system_error(errno, std::generic_category());
      }
      update--;
    }
  }

  static constexpr auto max() {
    return std::numeric_limits<unsigned int>::max();
  }

  sem_t sem;
};

#include "concurrentqueue.h"
#include "sample_cpu.hpp"
#include "utils.hpp"

using Adjs = std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                    std::pair<int64_t, int64_t>>>;
using ProtoSample = std::pair<torch::Tensor, Adjs>;
using PreparedSample = std::tuple<torch::Tensor, std::optional<torch::Tensor>,
                                  Adjs, std::pair<int32_t, int32_t>>;


//using ProtoDistributedBatch = std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor,
//                                  Adjs, std::pair<int32_t, int32_t>>;

// Converting to a struct for easier analysis on the c++ side with access to member variables.
struct ProtoDistributedBatch {
  std::vector<torch::Tensor> partition_nids;
  torch::Tensor sliced_cpu_features;
  torch::Tensor sliced_cpu_labels;
  torch::Tensor cached_nids;
  torch::Tensor perm_partition_to_mfg;
  Adjs adjs;
  std::pair<int32_t, int32_t> idx_range;
};


ProtoSample multilayer_sample(std::vector<int64_t> _n_ids,
                              std::vector<int64_t> const& sizes,
                              torch::Tensor rowptr, torch::Tensor col,
                              bool pin_memory = false) {
  //_tm105.start();
  std::vector<int32_t> n_ids(_n_ids.size());
  for (int64_t i = 0; i < n_ids.size(); i++) {
     n_ids[i] = (int32_t) (_n_ids[i]); 
  }
  //_tm105.stop();
  //_tm106.start();
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);
  Adjs adjs;
  adjs.reserve(sizes.size());
  //_tm106.stop();
  //_tm107.start();
  for (auto size : sizes) {
    auto const subset_size = n_ids.size();
    torch::Tensor out_rowptr, out_col, out_e_id;
    std::tie(out_rowptr, out_col, n_ids, out_e_id) = sample_adj(
        rowptr, col, std::move(n_ids), n_id_map, size, false, pin_memory);
    adjs.emplace_back(std::move(out_rowptr), std::move(out_col),
                      std::move(out_e_id),
                      std::make_pair(subset_size, n_ids.size()));
  }
  //_tm107.stop();
  //_tm108.start();

  _n_ids.resize(n_ids.size());
  for (int64_t i = 0; i < n_ids.size(); i++) {
      _n_ids[i] = (int64_t) n_ids[i];
  }

  std::reverse(adjs.begin(), adjs.end());
  //_tm108.stop();
  return {vector_to_tensor(_n_ids), std::move(adjs)};
}

ProtoSample multilayer_sample(torch::Tensor idx,
                              std::vector<int64_t> const& sizes,
                              torch::Tensor rowptr, torch::Tensor col,
                              bool pin_memory = false) {
  const auto idx_data = idx.data_ptr<int64_t>();
  return multilayer_sample({idx_data, idx_data + idx.numel()}, sizes,
                           std::move(rowptr), std::move(col), pin_memory);
}

template <typename scalar_t>
torch::Tensor serial_index_impl(torch::Tensor const in, torch::Tensor const idx,
                                int64_t const n,
                                bool const pin_memory = false) {
  const auto f = in.sizes().back();
  TORCH_CHECK((in.strides().size() == 2 && in.strides().back() == 1) ||
                  (in.sizes().back() == 1),
              "input must be 2D row-major tensor");

  torch::Tensor out =
      torch::empty({n, f}, in.options().pinned_memory(pin_memory));
  auto inptr = in.data_ptr<scalar_t>();
  auto outptr = out.data_ptr<scalar_t>();
  auto idxptr = idx.data_ptr<int64_t>();

  for (int64_t i = 0; i < std::min(idx.numel(), n); ++i) {
    const auto row = idxptr[i];
    std::copy_n(inptr + row * f, f, outptr + i * f);
  }

  return out;
}

template <typename scalar_t>
torch::Tensor serial_index_impl(torch::Tensor const in, torch::Tensor const idx,
                                bool const pin_memory = false) {
  return serial_index_impl<scalar_t>(in, idx, idx.numel(), pin_memory);
}

torch::Tensor serial_index(torch::Tensor const in, torch::Tensor const idx,
                           int64_t const n, bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, in.scalar_type(), "serial_index",
      [&] { return serial_index_impl<scalar_t>(in, idx, n, pin_memory); });
}

torch::Tensor serial_index(torch::Tensor const in, torch::Tensor const idx,
                           bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, in.scalar_type(), "serial_index",
      [&] { return serial_index_impl<scalar_t>(in, idx, pin_memory); });
}

torch::Tensor to_row_major(torch::Tensor const in) {
  TORCH_CHECK(in.strides().size() == 2, "only support 2D tensors");
  auto const tr = in.sizes().front();
  auto const tc = in.sizes().back();

  if (in.strides().front() == tc && in.strides().back() == 1) {
    return in;  // already in row major
  }

  TORCH_CHECK(in.strides().front() == 1 && tr == in.strides().back(),
              "input has unrecognizable stides");

  auto out = torch::empty_strided(in.sizes(), {tc, 1}, in.options());

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, in.scalar_type(),
                                 "to_row_major", [&] {
                                   auto inptr = in.data_ptr<scalar_t>();
                                   auto outptr = out.data_ptr<scalar_t>();

                                   for (int64_t r = 0; r < tr; ++r) {
                                     for (int64_t c = 0; c < tc; ++c) {
                                       outptr[r * tc + c] = inptr[c * tr + r];
                                     }
                                   }
                                 });

  return out;
}

template <typename x_scalar_t, typename y_scalar_t>
std::vector<std::vector<PreparedSample>> full_sample_impl(
    torch::Tensor const x, torch::Tensor const y, torch::Tensor const rowptr,
    torch::Tensor const col, torch::Tensor const idx, int64_t const batch_size,
    std::vector<int64_t> sizes, bool const skip_nonfull_batch = false,
    bool const pin_memory = false) {
  CHECK_CPU(x);
  CHECK_CPU(y);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);

  std::vector<std::vector<PreparedSample>> results(omp_get_max_threads());

  auto const n = idx.numel();
  const auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for schedule(dynamic)
  for (int64_t i = 0; i < n; i += batch_size) {
    auto const this_batch_size = std::min(n, i + batch_size) - i;
    if (skip_nonfull_batch && (this_batch_size < batch_size)) {
      continue;
    }

    const std::pair<int32_t, int32_t> pair = {i, i + this_batch_size};
    auto proto =
        multilayer_sample({idx_data + pair.first, idx_data + pair.second},
                          sizes, rowptr, col, pin_memory);
    auto const& n_id = proto.first;
    auto x_s = serial_index_impl<x_scalar_t>(x, n_id, pin_memory);
    auto y_s =
        serial_index_impl<y_scalar_t>(y, n_id, this_batch_size, pin_memory);
    results[omp_get_thread_num()].emplace_back(std::move(x_s), std::move(y_s),
                                               std::move(proto.second),
                                               std::move(pair));
  }

  return results;
}

std::vector<std::vector<PreparedSample>> full_sample(
    torch::Tensor const x, torch::Tensor const y, torch::Tensor const rowptr,
    torch::Tensor const col, torch::Tensor const idx, int64_t const batch_size,
    std::vector<int64_t> sizes, bool const skip_nonfull_batch = false,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, x.scalar_type(), "full_sample_x", [&] {
        using x_scalar_t = scalar_t;
        return AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half, y.scalar_type(), "full_sample_y", [&] {
              using y_scalar_t = scalar_t;
              return full_sample_impl<x_scalar_t, y_scalar_t>(
                  x, y, rowptr, col, idx, batch_size, sizes, skip_nonfull_batch,
                  pin_memory);
            });
      });
}

template <class Slot>
class Thread {
 public:
  template <class Function>
  Thread(Function&& f, std::unique_ptr<Slot> slot_)
      : slot{std::move(slot_)},
        thread{std::forward<Function>(f), std::ref(*(slot.get()))} {}

  template <class Function>
  Thread(Function&& f)
      : Thread{std::forward<Function>(f), std::make_unique<Slot>()} {}

  Thread(Thread&&) = default;
  Thread& operator=(Thread&&) = default;

  ~Thread() {
    if (slot) {
      slot->decommission();
      while (!thread.joinable()) {
        std::this_thread::yield();
      }
      thread.join();
    }
  }

  // TODO: Hide this behind getter.
  std::unique_ptr<Slot> slot;
  std::thread thread;
};

template <class Thread>
class ThreadPool {
 public:
  ThreadPool(std::function<Thread()> thread_factory_, size_t max_size)
      : thread_factory{std::move(thread_factory_)}, max_size{max_size} {}
  ThreadPool(std::function<Thread()> thread_factory_)
      : ThreadPool{std::move(thread_factory_),
                   std::numeric_limits<size_t>::max()} {}

  ~ThreadPool() {
    // Give notice ahead of time, to speed up shutdown
    for (auto& thread : pool) {
      thread.slot->decommission();
    }
  }

  // TODO: Make a container type that returns its threads to this pool on
  // destruction.
  std::vector<Thread> get(const size_t num) {
    std::vector<Thread> out;
    out.reserve(num);
    out.insert(out.end(),
               std::make_move_iterator(pool.end() - std::min(num, pool.size())),
               std::make_move_iterator(pool.end()));
    pool.erase(pool.end() - out.size(), pool.end());

    while (out.size() < num) {
      out.emplace_back(thread_factory());
    }
    return out;
  }

  void consume(std::vector<Thread> threads) {
    for (auto& thread : threads) {
      thread.slot->hibernate_begin();
    }

    // wait for successful hibernate
    for (auto& thread : threads) {
      thread.slot->hibernate_end();
    }

    pool.insert(pool.end(), std::make_move_iterator(threads.begin()),
                std::make_move_iterator(
                    threads.begin() +
                    std::min(max_size - pool.size(), threads.size())));
    // the remaining threads should get destructed
  }

 public:
  const std::function<Thread()> thread_factory;
  const size_t max_size;

 private:
  std::vector<Thread> pool;
};

class FastSamplerSession;
struct FastSamplerSlot {
  std::mutex mutex;
  std::condition_variable cv;

  FastSamplerSession* session = nullptr;
  moodycamel::ProducerToken optok{};
  moodycamel::ProducerToken optok_distributed{};

  // TODO: should these change to relaxed atomics?
  // we're willing to accept the thread trying to unsuccessfully dequeue
  // until the values propagate between cores
  // according to godbolt, both x86 gcc and clang are dumb
  // and make unnecessary test or and instructions when working with the atomic
  // The PowerPC instructions just look very funny in either case.
  // https://godbolt.org/z/sEYTcGP5o
  volatile bool hibernate_flag = true;
  volatile bool decommissioned = false;

  bool should_hibernate() const { return hibernate_flag; }

  bool should_decommission() const { return decommissioned; }

  void assign_session(FastSamplerSession& new_session);

  void hibernate_begin() {
    // stores are not reordered with stores
    // TODO: Specify memory order to make sure.
    hibernate_flag = true;
    sem_wake();
  }

  void hibernate_end() {
    const std::lock_guard<decltype(mutex)> lock(mutex);
    session = nullptr;
  }

  void decommission() {
    decommissioned = true;
    cv.notify_one();
    sem_wake();
  }

 public:
   // CACHING COMPONENT
   // Each thread will keep a frequency count for vertices on remote machine seen across all of its batches.
   //   The frequency will then be reduced across all threads after all batches on all threads have been processed.
   // NOTE: uint32_t may be sufficient for frequency.
   phmap::flat_hash_map<int64_t, int64_t> thread_local_remote_frequency_map;

 private:
  void sem_wake();
};

using FastSamplerThread = Thread<FastSamplerSlot>;

FastSamplerThread thread_factory();
ThreadPool<FastSamplerThread> global_threadpool{
    thread_factory, std::thread::hardware_concurrency()};

struct FastSamplerConfig {
  size_t batch_size;
  torch::Tensor x_cpu;
  torch::Tensor x_gpu;
  std::optional<torch::Tensor> y;
  torch::Tensor rowptr, col, idx;
  std::vector<int64_t> sizes;
  bool skip_nonfull_batch;
  bool pin_memory;
  bool distributed;
  RangePartitionBook partition_book;
  Cache cache;
  bool force_exact_num_batches;
  size_t exact_num_batches;
  bool count_remote_frequency;
  bool use_cache;
};

class FastSamplerSession {
 public:
  using Range = std::pair<int32_t, int32_t>;
  // using Chunk = std::pair<uint8_t, std::array<Range, 5>>;

  

  FastSamplerSession(size_t num_threads, unsigned int max_items_in_queue,
                     FastSamplerConfig config_)
      : config{std::move(config_)},
        // TODO: Why doesn't this compile...
        // threads{global_threadpool.get(num_threads)},
        items_in_queue{std::min(max_items_in_queue, items_in_queue.max())},
        // No need to enforce with exact_num_batches, init ConcurrentQueues with an initial size estimate, can be resized.
        inputs{config.idx.numel() / config.batch_size + 1},
        outputs{config.idx.numel() / config.batch_size + 1},
        outputs_distributed{config.idx.numel() / config.batch_size + 1},
        iptok{inputs},
        octok{outputs},
        octok_distributed{outputs_distributed} {
    //_tm1.reset();
    //_tm2.reset();
    //_tm3.reset();
    //_tm4.reset();
    //_tm5.reset();
    //_tm6.reset();
    //_tm7.reset();
    //_tm8.reset();
    //_tm9.reset();
    //_tm10.reset();
    //printf("Fast sampler session with num workers %d\n", num_threads);
    //_tm101.reset();
    //_tm102.reset();
    //_tm103.reset();
    //_tm104.reset();
    //_tm105.reset();
    //_tm106.reset();
    //_tm107.reset();
    //_tm108.reset();
    TORCH_CHECK(max_items_in_queue > 0,
                "max_items_in_queue (%llu) must be positive",
                max_items_in_queue);
    TORCH_CHECK(max_items_in_queue <= items_in_queue.max(),
                "max_items_in_queue (%llu) must be <= %ll", max_items_in_queue,
                items_in_queue.max());
    threads = global_threadpool.get(num_threads);
    for (FastSamplerThread& thread : threads) {
      thread.slot->assign_session(*this);
    }


    startTime()


    size_t const n = config.idx.numel();
    // In the distributed context, want to make sure that all machines run for the exact same number of iterations.
    // Otherwise, 1 or more machines may complete while the other ones hang in the all2all trying to exchange data.
    // Minor edge case: fails if avg_size < 1.
    // This branch ignores the batch_size.
    if (config.force_exact_num_batches) {
        std::vector<uint64_t> batch_sizes(config.exact_num_batches);
        int64_t remaining_elements_to_allocate = n;
        uint64_t avg_size = n / config.exact_num_batches - 1;
        for (size_t i = 0; i < config.exact_num_batches; i++) {
            batch_sizes[i] = avg_size;
            remaining_elements_to_allocate -= avg_size;
            assert(remaining_elements_to_allocate >= 0 && "remaining_elements_to_allocate >= 0 failed");
        }
        // maybe overkill :)
        while (remaining_elements_to_allocate > 0) {
            for (size_t i = 0; i < config.exact_num_batches; i++) {
                if (remaining_elements_to_allocate <= 0) break;
                batch_sizes[i]++;
                remaining_elements_to_allocate--;
            }
        }
        uint64_t sum = 0;
        for (size_t i = 0; i < config.exact_num_batches; i++) {
            num_total_batches++;
            inputs.enqueue(iptok, Range(sum, sum + batch_sizes[i]));
            sum += batch_sizes[i];
        }
        assert(sum == n && "sum == n failed");
    // This brach uses the batch_size.
    } else {
        for (size_t i = 0; i < n; i += config.batch_size) {
          auto const this_batch_size = std::min(n, i + config.batch_size) - i;
          if (config.skip_nonfull_batch && (this_batch_size < config.batch_size)) {
            continue;
          }

          num_total_batches++;
          inputs.enqueue(iptok, Range(i, i + this_batch_size));
        }
   }
  }

  ~FastSamplerSession() {

    //_tm1.reportTotal("map insert");
    //_tm2.reportTotal("N-id append");
    //_tm3.reportTotal("Body of addNeighbor");
    //_tm5.reportTotal("alloc tensors");
    //_tm4.reportTotal("postprocessing");
    //_tm6.reportTotal("actual time in robert floyd");
    //_tm7.reportTotal("entire interior body of multilayer sample");
    //_tm8.reportTotal("case 1 RF alg");
    //_tm9.reportTotal("case 2 RF alg");
    //_tm10.reportTotal("addNeighbor");
    //_tm101.reportTotal("multilayer sampling");
    //_tm102.reportTotal("all before slicing");
    //_tm103.reportTotal("slicing");
    //_tm104.reportTotal("all after slicing");
    //_tm105.reportTotal("ML convert to 32");
    //_tm106.reportTotal("get initial adj hash map");
    //_tm107.reportTotal("do multilayer sample");
    //_tm108.reportTotal("convert back to 64");
    // Return the threads to the pool.
    global_threadpool.consume(std::move(threads));
  }
  size_t idx_range_start = 0;
  size_t buffer_size = 0;
  bool buffer_init = false;
  std::vector<ProtoDistributedBatch> buffer;

  std::optional<PreparedSample> try_get_batch() {
    if (num_consumed_batches == num_total_batches) {
      return {};
    }

    PreparedSample batch;
    if (!outputs.try_dequeue(octok, batch)) {
      return {};
    }
    num_consumed_batches++;
    items_in_queue.release();
    return batch;
  }

  std::optional<ProtoDistributedBatch> try_get_batch_distributed() {
    if (!buffer_init) {
       buffer_init = true;
       buffer = std::vector<ProtoDistributedBatch>(200);
    }
    if (num_consumed_batches == num_total_batches) {
      //std::cout << "num_consumed_batches == num_total_batches" << std::endl;
      return {};
    }
    //std::cout << "approx size: " << outputs_distributed.size_approx() << std::endl;

    ProtoDistributedBatch batch;
    if (!outputs_distributed.try_dequeue(octok_distributed, batch)) {
      //std::cout << "!outputs_distributed.try_dequeue(octok_distributed, batch)" << std::endl;


      //return {};
    } else {
      buffer[buffer_size++] = std::move(batch);
    }
    ProtoDistributedBatch ret;
    bool found = false;
    for (int i = 0; i < buffer_size; i++) {
      if (buffer[i].idx_range.first == idx_range_start) {
          idx_range_start = buffer[i].idx_range.second;
          buffer_size--;
          ret = std::move(buffer[i]);
          buffer[i] = std::move(buffer[buffer_size]);
          found = true;
          break;
      }
    } 
    if (!found) {
      return {};
    } else  {
      //std::cout << "should be working" << std::endl;
      num_consumed_batches++;
      items_in_queue.release();
      return ret;
    }
  }



  std::future<std::vector<std::vector<torch::Tensor>>> async_slice_future;
  std::vector<std::vector<torch::Tensor>> async_slice_future_result;
  bool has_async_future_result = false; 
 
  void async_slice_tensors (std::vector<torch::Tensor> ids, int my_rank) {
     assert(has_async_future_result == false && "has_async_future_result == false\n");
     async_slice_future = std::async(std::launch::async, [ids, this, my_rank]() {
         std::vector<std::vector<torch::Tensor> > results(ids.size());
         for (int i = 0; i < ids.size(); i++) {
             //if (i == my_rank) continue;
            
             std::vector<int64_t> local_on_cpu;

             std::vector<int64_t> cpu_pos;
             std::vector<int64_t> gpu_pos;
             cpu_pos.reserve(ids[i].numel());
             gpu_pos.reserve(ids[i].numel());
             for (int j = 0; j < ids[i].numel(); j++) {
                 if (ids[i].data_ptr<int64_t>()[j] >= 0) {
                     local_on_cpu.push_back(ids[i].data_ptr<int64_t>()[j]);
                     cpu_pos.push_back(j);//ids[i].data_ptr<int64_t>()[j]);
                 } else {
                     //local_on_gpu.push_back();
                     gpu_pos.push_back(j);//ids[i].data_ptr<int64_t>()[j]);
                 }
             }
 
             auto local_on_cpu_tensor = vector_to_tensor(local_on_cpu);
             auto cpu_pos_tensor = vector_to_tensor(cpu_pos, true);
             auto gpu_pos_tensor = vector_to_tensor(gpu_pos, true);
             if (i != my_rank) { 
                 auto x_s = serial_index(this->config.x_cpu, local_on_cpu_tensor, true);
                 results[i] = std::vector<torch::Tensor>{std::move(x_s), std::move(cpu_pos_tensor), std::move(gpu_pos_tensor)};
             } else {
                 auto x_s = torch::empty(0, torch::TensorOptions().dtype(torch::kInt64));
                 results[i] = std::vector<torch::Tensor>{std::move(x_s), std::move(cpu_pos_tensor), std::move(gpu_pos_tensor)};
             }
         }
         return results;
     });
     has_async_future_result = true;
     return; 
  }

  void wait_slice_tensors() {
      if (!has_async_future_result) {
         //printf("Warning no async future result.\n");
         //fflush(stdout);
         return;
      }
      assert(has_async_future_result && "has_async_future_result\n"); 
      async_slice_future.wait();
      async_slice_future_result = async_slice_future.get();
      has_async_future_result = false;
      return;
  }

  std::vector<std::vector<torch::Tensor> > get_slice_tensors () {
     return async_slice_future_result;
  }


  std::optional<PreparedSample> blocking_get_batch() {
    if (num_consumed_batches == num_total_batches) {
      return {};
    }

    auto batch = try_get_batch();
    if (batch) {
      return batch;
    }

    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
      auto batch = try_get_batch();
      if (batch) {
        auto end = std::chrono::high_resolution_clock::now();
        total_blocked_dur +=
            std::chrono::duration_cast<decltype(total_blocked_dur)>(end -
                                                                    start);
        total_blocked_occasions++;
        return batch;
      }
    }
  }

  std::optional<ProtoDistributedBatch> blocking_get_batch_distributed() {
    if (num_consumed_batches == num_total_batches) {
      return {};
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto batch = try_get_batch_distributed();
    if (batch) {
      auto end = std::chrono::high_resolution_clock::now();
      total_blocked_dur +=
            std::chrono::duration_cast<decltype(total_blocked_dur)>(end - start);

      return batch;
    }

    while (true) {
      auto batch = try_get_batch_distributed();
      if (batch) {
        auto end = std::chrono::high_resolution_clock::now();
        total_blocked_dur +=
            std::chrono::duration_cast<decltype(total_blocked_dur)>(end -
                                                                    start);
        total_blocked_occasions++;
        return batch;
      }
    }
  }

  // CACHING COMPONENT
  // reduce the thread_local_remote_frequency_map for all threads into remote_frequency_map
  // After collecting frequency data in a map, a tensor is desirable for sorting and sending data over to the python side.
  // Need a pair of tensors to store the vertices and corresponding frequencies.
  // For convenience, the frequency and corresponding vertex data will be sorted to be ordered by frequency.
  void reduce_multithreaded_frequency_counts() {
    if (this->remote_frequency_reduced) {
        return;
    }
    for (FastSamplerThread& thread : threads) {
      phmap::flat_hash_map<int64_t, int64_t> thread_local_remote_frequency_map = thread.slot->thread_local_remote_frequency_map;
      for(auto& element : thread_local_remote_frequency_map) {
       int64_t vertex = element.first;
       int64_t frequency = element.second;
       this->remote_frequency_map[vertex] += frequency;
      }
    }
    // Initially when these tensors are created they will not be sorted.
    int64_t num_remote_vertices = this->remote_frequency_map.size();
    this->remote_frequency_tensor = torch::empty(num_remote_vertices, torch::TensorOptions().dtype(torch::kInt64));
    this->remote_vertices_ordered_by_freq = torch::empty(num_remote_vertices, torch::TensorOptions().dtype(torch::kInt64));
    auto remote_frequency_tensor_data = this->remote_frequency_tensor.data_ptr<int64_t>();
    auto remote_vertices_ordered_by_freq_data = this->remote_vertices_ordered_by_freq.data_ptr<int64_t>();
    int64_t i = 0;
    for(auto& element : this->remote_frequency_map) {
     int64_t vertex = element.first;
     int64_t frequency = element.second;
     remote_frequency_tensor_data[i] = frequency;
     remote_vertices_ordered_by_freq_data[i] = vertex;
     i++;
    }

    // Sort descending.
    torch::Tensor sorted_indices = at::argsort(remote_frequency_tensor, -1, true);
    this->remote_frequency_tensor = at::index_select(this->remote_frequency_tensor, 0, sorted_indices);
    this->remote_vertices_ordered_by_freq = at::index_select(this->remote_vertices_ordered_by_freq, 0, sorted_indices);
    
    // Update reduced flag;
    this->remote_frequency_reduced = true;
  }

  // CACHING COMPONENT
  // return the n most frequently accessed remote vertices.
  torch::Tensor get_n_most_freq_remote_vertices(int n) {
    this->reduce_multithreaded_frequency_counts();
    torch::Tensor out = torch::empty(n, torch::TensorOptions().dtype(torch::kInt64));
    auto out_ptr = out.data_ptr<int64_t>();
    auto remote_vertices_ordered_by_freq_ptr = this->remote_vertices_ordered_by_freq.data_ptr<int64_t>();
    std::copy_n(remote_vertices_ordered_by_freq_ptr, n, out_ptr);
    return out;
  }

  size_t get_num_consumed_batches() const { return num_consumed_batches; }

  size_t get_approx_num_complete_batches() const {
    return num_consumed_batches + num_inserted_batches;
  }

  size_t get_num_total_batches() const { return num_total_batches; }

  torch::Tensor get_remote_frequency_tensor() const { return remote_frequency_tensor; }
  torch::Tensor get_remote_vertices_ordered_by_freq() const { return remote_vertices_ordered_by_freq; }

  const FastSamplerConfig config;

  std::atomic<size_t> num_inserted_batches{0};

 private:
  std::vector<FastSamplerThread> threads;
  size_t num_consumed_batches = 0;
  size_t num_total_batches = 0;

 public:
  // std::counting_semaphore<> items_in_queue;
  MySemaphore items_in_queue;

 public:
  moodycamel::ConcurrentQueue<Range> inputs;
  moodycamel::ConcurrentQueue<PreparedSample> outputs;
  moodycamel::ConcurrentQueue<ProtoDistributedBatch> outputs_distributed;
  moodycamel::ProducerToken iptok;  // input producer token
  moodycamel::ConsumerToken octok;  // output consumer token





  // output consumer token for the discerning distributed programmer.
  moodycamel::ConsumerToken octok_distributed;  

  // benchmarking data
  std::chrono::microseconds total_blocked_dur{};
  size_t total_blocked_occasions = 0;

  // CACHING COMPONENT
  // Hashmap to reduce across all threads after all batches on all threads have been processed.
  // NOTE: uint32_t may be sufficient for frequency.
  phmap::flat_hash_map<int64_t, int64_t> remote_frequency_map;
  // After collecting frequency data in a map, a tensor is desirable for sorting and sending data over to the python side.
  // Need a pair of tensors to store the vertices and corresponding frequencies.
  // For convenience, the frequency and corresponding vertex data will be sorted to be ordered by frequency.
  torch::Tensor remote_frequency_tensor;
  torch::Tensor remote_vertices_ordered_by_freq;
  // Flag whether or not the reduce has happened yet;
  bool remote_frequency_reduced = false;

};

void FastSamplerSlot::assign_session(FastSamplerSession& new_session) {
  std::unique_lock<decltype(mutex)> lock(mutex);
  session = &new_session;
  optok = moodycamel::ProducerToken{new_session.outputs};
  optok_distributed = moodycamel::ProducerToken{new_session.outputs_distributed};

  hibernate_flag = false;
  lock.unlock();
  cv.notify_one();
}

void FastSamplerSlot::sem_wake() {
  if (session == nullptr) {
    return;
  }
  session->items_in_queue.release();
}

void print_tensor_int64(torch::Tensor t, std::string name) {
    std::cout << name << std::endl;
    std::for_each(t.data_ptr<int64_t>(), t.data_ptr<int64_t>() + t.numel(),
        [](int64_t v){ std::cout << std::setw(2) << v <<  ' '; });
    std::cout << std::endl;
}

void fast_sampler_thread(FastSamplerSlot& slot) {
  std::unique_lock<decltype(slot.mutex)> lock(slot.mutex);
  while (true) {
    if (slot.should_hibernate()) {
      slot.cv.wait(lock, [&slot] {
        return slot.should_decommission() || !slot.should_hibernate();
      });
    }

    if (slot.should_decommission()) {
      return;
    }

    std::pair<int32_t, int32_t> pair;
    if (!slot.session->inputs.try_dequeue_from_producer(slot.session->iptok,
                                                        pair)) {
      ;
      continue;
    }

    slot.session->items_in_queue.acquire();

    // check if we were woken just to decommission or hibernate
    if (slot.should_hibernate() || slot.should_decommission()) {
      continue;
    }

    auto const this_batch_size = pair.second - pair.first;

    auto const& config = slot.session->config;
    const auto idx_data = config.idx.data_ptr<int64_t>();
    gen.seed(pair.second * 17 + 5);
    nextTime("before multilayer_sample")

    //_tm101.start();
    auto proto = multilayer_sample(
        {idx_data + pair.first, idx_data + pair.second}, config.sizes,
        config.rowptr, config.col, config.pin_memory);
    //_tm101.stop();
    nextTime("after multilayer_sample")

    if (!config.distributed) {
      auto const& n_id = proto.first;
      auto x_s = serial_index(config.x_cpu, n_id, config.pin_memory);
      std::optional<torch::Tensor> y_s;
      if (config.y.has_value()) {
        y_s = serial_index(*config.y, n_id, this_batch_size, config.pin_memory);
      }

      slot.session->outputs.enqueue(
          slot.optok, {std::move(x_s), std::move(y_s), std::move(proto.second),
                       std::move(pair)});
      ++slot.session->num_inserted_batches;
      continue;
    } else {

      _tm102.start();
      auto const& n_id = proto.first;
      auto partition_book = config.partition_book;

      std::vector<torch::Tensor> partition_nids(partition_book.world_size);
      //Allocate the inverse_permutation tensor: a tensor that if the partitioned vertex ids (and potentially cached vertices) are concatenated,
      //indexing with this tensor would return a tensor of vertex ids in the original order in the n_id tensor.
      torch::Tensor inverse_permutation = torch::empty(n_id.numel(), n_id.options().pinned_memory(true));
      torch::Tensor cached_nids;
      torch::Tensor x_s,y_s;

      // CACHING COMPONENT
      if (!config.use_cache) { 
      if (config.x_cpu.sizes().back() > 0) {
          // Identify which vertices are in the local partition.
          torch::Tensor local_bool = partition_book.nid_is_local(n_id);
          torch::Tensor local_indices = local_bool.nonzero().view(-1);
          torch::Tensor remote_indices = (~local_bool).nonzero().view(-1);
          torch::Tensor local = torch::empty(local_indices.numel(), torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
          torch::Tensor remote = torch::empty(remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64));
          local = n_id.index({local_indices}); 
          // slice the local indices that are not on-GPU
          std::vector<int64_t> local_on_cpu;
          local_on_cpu.reserve(local.numel());
          torch::Tensor local_local_ids = partition_book.nid2localnid(local, partition_book.rank);
          const int64_t x_gpu_size = config.x_gpu.sizes()[0];
           
          for (int i = 0; i < local_local_ids.numel(); i++) {
             if (local_local_ids.data_ptr<int64_t>()[i] >= x_gpu_size) {
                 local_on_cpu.push_back(local_local_ids.data_ptr<int64_t>()[i] - x_gpu_size);
             }
          }
          auto local_on_cpu_tensor = vector_to_tensor(local_on_cpu);
          x_s = serial_index(config.x_cpu, local_on_cpu_tensor, config.pin_memory);
          if (config.y.has_value()) {
            y_s = serial_index(*config.y, n_id, this_batch_size, config.pin_memory);
          }
      } else {
          x_s = torch::zeros(0);
          y_s = torch::zeros(0);
      }


 
        torch::Tensor machine_id = partition_book.nid2partid(n_id);
        // Count how many vertex ids belong to each partition.
        torch::Tensor partition_bincount = torch::bincount(machine_id, {}, partition_book.world_size);
        // Using data_ptrs, could also explicitly use torch::indexing.
        auto n_id_data = n_id.data_ptr<int64_t>();
        auto machine_id_data = machine_id.data_ptr<int64_t>();
        auto partition_bincount_data = partition_bincount.data_ptr<int64_t>();
        // Update the vector of tensors to store the partitioned vertex ids.
        for (int i = 0; i < partition_bincount.numel(); i++) {
          partition_nids[i] = torch::empty(partition_bincount_data[i], n_id.options().pinned_memory(true));
        }
        // Calculate the prefix sum of the partition bincout.
        std::vector<int64_t> partition_offsets(partition_nids.size()+1, 0);
        for (int i = 0; i < partition_bincount.numel(); i++) {
          partition_offsets[i+1] = partition_offsets[i] + partition_bincount_data[i];
        }
        // Keep a count for each partition.
        // Create the inverse mappping.
        std::vector<int64_t> counts(partition_nids.size(), 0);
        for (int i = 0; i < n_id.numel(); i++) {
           int m = machine_id_data[i];
           partition_nids[m].data_ptr<int64_t>()[counts[m]] = n_id_data[i];
           inverse_permutation.data_ptr<int64_t>()[i] = partition_offsets[m]+counts[m];
           counts[m]++;
        }

        // CACHING COMPONENT
        // This section of the function records the frequency at which a feature for a vertex not on the partition is requried.
        // As earlier in this function have determined which vertices are on the local partition, iterate through vertices
        //    on other partitions update their frequency of appearance.
        if (config.count_remote_frequency) {
          int local_rank = config.partition_book.rank;
          int num_parts = config.partition_book.world_size;
          for (int rank = 0; rank < num_parts; rank++) {
            if (rank != local_rank) {
              for (int i = 0; i < partition_nids[rank].numel(); i++) {
                slot.thread_local_remote_frequency_map[partition_nids[rank].data_ptr<int64_t>()[i]]++;
              }
            }
          }
        }

        // Not doing any caching here, so return an empty tensor.
        cached_nids = torch::empty(0, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

      } else {

        nextTime("everything_else")
        /* 
        Using cache.  
        Simple approach:
          1. Identify which vertices are in local partition.
          2. Identify which vertices not in local partition are cached.
          3. Identify partitions of vertices not in local partition and not cached.
          4. While doing the previous 2 steps, (for world size == k) construct an inverse permutation
             assuming we concatenate the features corresponding to the vertices like so: 
               1. partititon_nids[0]
               2. partititon_nids[1]
               ..
               k. partition_nids[k-1]
               k+1. cached (remote vertices in cache)

               All of the above vertex tensors and inverse permutation are pinned. 
        */

        // Identify which vertices are in the local partition.
        torch::Tensor local_bool = partition_book.nid_is_local(n_id);
        torch::Tensor local_indices = local_bool.nonzero().view(-1).to(torch::kInt64).contiguous();
        torch::Tensor remote_indices = (~local_bool).nonzero().view(-1);
        torch::Tensor local = torch::empty(local_indices.numel(), torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        torch::Tensor remote = torch::empty(remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64));

           
        torch::Tensor local_proto = n_id.index({local_indices});
        for (int64_t i = 0; i < local_proto.numel(); i++) {
           local.data_ptr<int64_t>()[i] = local_proto.data_ptr<int64_t>()[i];
        }

        // slice the local indices that are not on-GPU
        std::vector<int64_t> local_on_cpu;
        local_on_cpu.reserve(local.numel());
        const int64_t x_gpu_size = config.x_gpu.sizes()[0];
        torch::Tensor local_local_ids = partition_book.nid2localnid(local, partition_book.rank);
        for (int i = 0; i < local_local_ids.numel(); i++) {
           if (local_local_ids.data_ptr<int64_t>()[i] >= x_gpu_size) {
               local_on_cpu.push_back(local_local_ids.data_ptr<int64_t>()[i] - x_gpu_size);
           }
        }
        _tm102.stop();
        _tm103.start();
        //printf("number of indices %llu\n", local_on_cpu.size());
        auto local_on_cpu_tensor = vector_to_tensor(local_on_cpu);
        x_s = serial_index(config.x_cpu, local_on_cpu_tensor, config.pin_memory);
        y_s = serial_index(config.y.value(), n_id, this_batch_size, config.pin_memory);

        _tm103.stop();
        _tm104.start();
        remote = n_id.index({remote_indices}); 

        nextTime("block1")
        // Identify which remote vertices are cached.

        nextTime("block2a")
        //Cache& cache = (Cache)config.cache;
        nextTime("block2b")
        nextTime("block2b2")
        // Only check if remote vertices are cached.
        torch::Tensor cached_bool = config.cache.nid_is_cached(remote);
        nextTime("block2c")
        torch::Tensor cached_out_of_remote_indices = cached_bool.nonzero().view(-1);
        nextTime("block2d")
        torch::Tensor not_cached_out_of_remote_indices = (~cached_bool).nonzero().view(-1);
        nextTime("block2e")

        torch::Tensor cached = torch::empty(cached_out_of_remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        nextTime("block2f")
        torch::Tensor remote_not_cached = torch::empty(not_cached_out_of_remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64));
        nextTime("block2g")

        cached = remote.index({cached_out_of_remote_indices}); 
        nextTime("block2h")
        remote_not_cached = remote.index({not_cached_out_of_remote_indices}); 
        nextTime("block2i")
        auto remote_not_cached_data = remote_not_cached.data_ptr<int64_t>();
        nextTime("block2j")

        nextTime("block2")
        // Since remote_cached_indices and remote_not_cached indices are indices into the remote tensor,
        //   for the inverse permutation have to convert them into indices into the tensor of all vertices (n_id).
        torch::Tensor cached_indices = torch::empty(cached_out_of_remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64));
        torch::Tensor remote_not_cached_indices = torch::empty(not_cached_out_of_remote_indices.numel(), torch::TensorOptions().dtype(torch::kInt64));

        cached_indices = remote_indices.index({cached_out_of_remote_indices});
        remote_not_cached_indices = remote_indices.index({not_cached_out_of_remote_indices});
        auto remote_not_cached_indices_data = remote_not_cached_indices.data_ptr<int64_t>();

        nextTime("block3")
        // Identify partitions of vertices not in local partition and not cached.
        // Rewriting this section.
        torch::Tensor remote_not_cached_partition_id = partition_book.nid2partid(remote_not_cached);
        torch::Tensor remote_not_cached_partition_bincount = torch::bincount(remote_not_cached_partition_id, {}, partition_book.world_size);
        auto remote_not_cached_partition_id_data = remote_not_cached_partition_id.data_ptr<int64_t>();
        auto remote_not_cached_partition_bincount_data = remote_not_cached_partition_bincount.data_ptr<int64_t>();
        nextTime("block4")

        // Write to  vector of tensors to store the partitioned vertex ids.
        // partition_nids[i] for i == local_rank, partitions_nids[i] = local, the vertices owned by the partition.
        // partition_nids[i] for i != local_rank, are the vertices not owned by the partition and not in cache.
        // Create a vector of tensors to store the indices corresponding to partitioned vertex ids, used in creating the inverse permutation.
        std::vector<torch::Tensor> partition_nids_indices(partition_book.world_size);

        for (int rank = 0; rank < partition_book.world_size; rank++) {
          if (rank == partition_book.rank) {
            partition_nids[rank] = local;
            partition_nids_indices[rank] = local_indices;
          } else {
            partition_nids[rank] = torch::empty(remote_not_cached_partition_bincount_data[rank], n_id.options().pinned_memory(true));
            partition_nids_indices[rank] = torch::empty(remote_not_cached_partition_bincount_data[rank], n_id.options());
          } 
        }

        nextTime("block5")

        // Partition remote_not_cached vertices and corresponding indices by remote partition.
        std::vector<int64_t> counts(partition_book.world_size, 0);
        for (int i = 0; i < remote_not_cached.numel(); i++) {
           int m = remote_not_cached_partition_id_data[i];
           if (m == partition_book.rank) {
           }
           partition_nids[m].data_ptr<int64_t>()[counts[m]] = remote_not_cached_data[i];
           partition_nids_indices[m].data_ptr<int64_t>()[counts[m]] = remote_not_cached_indices_data[i];
           counts[m]++;
        }

        nextTime("block6")
        //Construct inverse mapping.
        // Create a vector for all indices, partition_nids_indices and cached indices
        // use the copy constructor for partition_nids_indices
        std::vector<torch::Tensor> all_indices(partition_nids_indices);
        // insert cached_indices 
        all_indices.push_back(cached_indices);
        // Concatenate into one tensor.
        // Called the flip tensor because there is one last step before it is the true inverse permutation.
        torch::Tensor flipped_inverse_permutation = torch::empty(n_id.numel(), n_id.options());
        at::concat_out(flipped_inverse_permutation, at::TensorList(all_indices));

        for (int64_t i = 0; i < n_id.numel(); i++) {
           int64_t v = flipped_inverse_permutation.data_ptr<int64_t>()[i];
           inverse_permutation.data_ptr<int64_t>()[v] = i;
        } 

        nextTime("block7")
        // For cached vertices, translate to indices into the local cache.
        cached_nids = config.cache.nid2cachenid(cached);
        nextTime("block8")
        nextTime("block9")

      }
      // TODO: Check if there are copies here when creating the optional.
      ProtoDistributedBatch out = {std::move(partition_nids), std::move(x_s), std::move(y_s), std::move(cached_nids), std::move(inverse_permutation), std::move(proto.second), std::move(pair)};
      std::optional<ProtoDistributedBatch> out_opt = out;
      slot.session->outputs_distributed.enqueue(out);
      _tm104.stop();
      //slot.session->outputs_distributed.enqueue(
                                   // RESOLVE move
      //    slot.optok_distributed, {std::move(partition_nids), std::move(cached_nids), std::move(inverse_permutation), std::move(proto.second),
      //                 std::move(pair)});
      ++slot.session->num_inserted_batches;
      
    }
  }
}

FastSamplerThread thread_factory() {
  return FastSamplerThread{fast_sampler_thread};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ProtoDistributedBatch>(m, "ProtoDistributedBatch")
      .def(py::init<>())
      .def_readwrite("partition_nids", &ProtoDistributedBatch::partition_nids)
      .def_readwrite("sliced_cpu_features", &ProtoDistributedBatch::sliced_cpu_features)
      .def_readwrite("sliced_cpu_labels", &ProtoDistributedBatch::sliced_cpu_labels)
      .def_readwrite("cached_nids", &ProtoDistributedBatch::cached_nids)
      .def_readwrite("perm_partition_to_mfg", &ProtoDistributedBatch::perm_partition_to_mfg)
      .def_readwrite("adjs", &ProtoDistributedBatch::adjs)
      .def_readwrite("idx_range", &ProtoDistributedBatch::idx_range);
  py::class_<FastSamplerConfig>(m, "Config")
      .def(py::init<>())
      .def_readwrite("x_cpu", &FastSamplerConfig::x_cpu)
      .def_readwrite("x_gpu", &FastSamplerConfig::x_gpu)
      .def_readwrite("y", &FastSamplerConfig::y)
      .def_readwrite("rowptr", &FastSamplerConfig::rowptr)
      .def_readwrite("col", &FastSamplerConfig::col)
      .def_readwrite("idx", &FastSamplerConfig::idx)
      .def_readwrite("batch_size", &FastSamplerConfig::batch_size)
      .def_readwrite("sizes", &FastSamplerConfig::sizes)
      .def_readwrite("skip_nonfull_batch",
                     &FastSamplerConfig::skip_nonfull_batch)
      .def_readwrite("pin_memory", &FastSamplerConfig::pin_memory)
      .def_readwrite("distributed", &FastSamplerConfig::distributed)
      .def_readwrite("partition_book", &FastSamplerConfig::partition_book)
      .def_readwrite("cache", &FastSamplerConfig::cache)
      .def_readwrite("force_exact_num_batches", &FastSamplerConfig::force_exact_num_batches)
      .def_readwrite("exact_num_batches", &FastSamplerConfig::exact_num_batches)
      .def_readwrite("count_remote_frequency", &FastSamplerConfig::count_remote_frequency)
      .def_readwrite("use_cache", &FastSamplerConfig::use_cache);
  py::class_<FastSamplerSession>(m, "Session")
      .def(py::init<size_t, unsigned int, FastSamplerConfig>(),
           py::arg("num_threads"), py::arg("max_items_in_queue"),
           py::arg("config"))
      .def_readonly("config", &FastSamplerSession::config)
      .def("try_get_batch", &FastSamplerSession::try_get_batch)
      .def("try_get_batch_distributed", &FastSamplerSession::try_get_batch_distributed)
      .def("get_slice_tensors", &FastSamplerSession::get_slice_tensors)
      .def("wait_slice_tensors", &FastSamplerSession::wait_slice_tensors)
      .def("async_slice_tensors", &FastSamplerSession::async_slice_tensors)
      .def("blocking_get_batch", &FastSamplerSession::blocking_get_batch)
      .def("blocking_get_batch_distributed", &FastSamplerSession::blocking_get_batch_distributed)
      .def("reduce_multithreaded_frequency_counts", &FastSamplerSession::reduce_multithreaded_frequency_counts)
      .def("get_n_most_freq_remote_vertices", &FastSamplerSession::get_n_most_freq_remote_vertices,
           py::arg("n"))
      .def_property_readonly("num_consumed_batches",
                             &FastSamplerSession::get_num_consumed_batches)
      .def_property_readonly("num_total_batches",
                             &FastSamplerSession::get_num_total_batches)
      .def_property_readonly(
          "approx_num_complete_batches",
          &FastSamplerSession::get_approx_num_complete_batches)
      .def_readonly("total_blocked_dur", &FastSamplerSession::total_blocked_dur)
      .def_readonly("total_blocked_occasions",
                    &FastSamplerSession::total_blocked_occasions)
      .def_property_readonly("remote_frequency_tensor",
          &FastSamplerSession::get_remote_frequency_tensor, py::return_value_policy::reference_internal)
      .def_property_readonly("remote_vertices_ordered_by_freq",
          &FastSamplerSession::get_remote_vertices_ordered_by_freq, py::return_value_policy::reference_internal);
  m.def("sample_adj",
        py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, int32_t,
                          bool, bool>(&sample_adj),
        "Sample the one-hop neighborhood of the batch nodes", py::arg("rowptr"),
        py::arg("col"), py::arg("idx"), py::arg("num_neighbors"),
        py::arg("replace"), py::arg("pin_memory") = false);
  m.def("multilayer_sample",
      py::overload_cast<torch::Tensor, std::vector<int64_t> const&,
                        torch::Tensor, torch::Tensor, bool>(&multilayer_sample),
      "Sample the multi-hop neighborhood of the batch nodes", py::arg("idx"),
      py::arg("sizes"), py::arg("rowptr"), py::arg("col"),
      py::arg("pin_memory") = false);
  m.def("full_sample", &full_sample,
        "Parallel sample of the index divided into batch_size chunks",
        py::arg("x"), py::arg("y"), py::arg("rowptr"), py::arg("col"),
        py::arg("idx"), py::arg("batch_size"), py::arg("sizes"),
        py::arg("skip_nonfull_batch") = false, py::arg("pin_memory") = false);
  m.def("to_row_major", &to_row_major, "Convert 2D tensor to row major");
  m.def("serial_index",
        py::overload_cast<torch::Tensor, torch::Tensor, bool>(&serial_index),
        "Extract the rows of in (2D) as specified by idx", py::arg("in"),
        py::arg("idx"), py::arg("pin_memory") = false);
  m.def("serial_index",
        py::overload_cast<torch::Tensor, torch::Tensor, int64_t, bool>(
            &serial_index),
        "Extract the rows of in (2D) as specified by idx, up to n rows",
        py::arg("in"), py::arg("idx"), py::arg("n"),
        py::arg("pin_memory") = false);

  py::class_<RangePartitionBook>(m, "RangePartitionBook")
      .def(py::init<size_t, size_t, torch::Tensor>(),
          py::arg("rank"), py::arg("world_size"), py::arg("partition_offsets"))
      .def_readwrite("rank", &RangePartitionBook::rank)
      .def_readwrite("world_size", &RangePartitionBook::world_size)
      .def_readwrite("partition_offsets", &RangePartitionBook::partition_offsets)
      .def("nid2localnid",
          &RangePartitionBook::nid2localnid,
          py::arg("nids"), py::arg("partition_idx"))
      .def("nid2partid",
          &RangePartitionBook::nid2partid,
          py::arg("nids"))
      .def("partid2nids",
          &RangePartitionBook::partid2nids,
          py::arg("partition_idx"));
  py::class_<Cache>(m, "Cache")
      .def(py::init<>())
      .def(py::init<size_t, size_t, torch::Tensor, torch::Tensor>(),
        py::arg("rank"), py::arg("world_size"), py::arg("cached_vertices"), py::arg("cached_features"))
      .def_readonly("rank", &Cache::rank)
      .def_readonly("world_size", &Cache::world_size)
      .def_readonly("cached_vertices", &Cache::cached_vertices)
      .def_readonly("cached_features", &Cache::cached_features)
      .def("nid_is_cached", &Cache::nid_is_cached,
          py::arg("nids"))
      .def("nid2cachenid", &Cache::nid2cachenid,
          py::arg("nids"));
    ;
}

