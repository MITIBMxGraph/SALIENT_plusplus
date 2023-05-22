#include "range_partition_book.hpp"
//#include <ATen/ATen.h>
//#include <ATen/core/TensorAccessor.h>
//#include <ATen/cuda/CUDAContext.h>
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
#define startTime() //_tm.start();
#define nextTime(_string) //_tm.next(_string);



RangePartitionBook::RangePartitionBook(
  int rank_, int world_size_, torch::Tensor partition_offsets_
) : rank{rank_}, world_size{world_size_}, partition_offsets{partition_offsets_} {};

torch::Tensor RangePartitionBook::nid2localnid(torch::Tensor nids, int partition_idx) const {
  //printf("current device of nids nids %d\n", nids.get_device());
  //printf("current device of partition_offsets  %d\n", this->partition_offsets.get_device());
  //std::cout << at::cuda::getCurrentCUDAStream() << std::endl;
  //std::cout << at::cuda::getCurrentCUDAStream(0) << std::endl;
    
  return nids - this->partition_offsets[partition_idx];
}

torch::Tensor RangePartitionBook::nid2partid(torch::Tensor nids) const {
 return at::searchsorted(this->partition_offsets, nids, false, true) - 1; 
}

// Returns a boolean tensor with each element either being an element if in the local partition, 0 if not.
// Not that it matters but curious if this && short-circuits.
// Come back if want an and operator, multiplying booleans should be equivalent.
torch::Tensor RangePartitionBook::nid_is_local(torch::Tensor nids) const {
 return (nids >= this->partition_offsets[rank])*(nids < this->partition_offsets[rank+1]); 
}

torch::Tensor RangePartitionBook::partid2nids(int partition_idx) const {
  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  return torch::arange(this->partition_offsets[partition_idx].item(), this->partition_offsets[partition_idx + 1].item(), opts);
}


// The default constructor initializes the member tensors to empty, otherwise pybind will return them as NoneType.
Cache::Cache() {
  this->cached_vertices = torch::empty(0, torch::TensorOptions().dtype(torch::kInt64));
  this->cached_features = torch::empty((0,0), torch::TensorOptions().dtype(torch::kFloat16));
}

Cache::Cache(
  int rank_,
  int world_size_,
  torch::Tensor cached_vertices_,
  torch::Tensor cached_features_)
  : rank{rank_}, world_size{world_size_},
    cached_vertices{cached_vertices_}, cached_features{cached_features_}
{
  // create the cached_vertices_map
  auto cached_vertices_data = this->cached_vertices.data_ptr<int64_t>();
  // Welcome intrepid code speluncker, it seems you (like I) have stumbled upon a most
  // curious piece of code. Well there is a tale behind this 'hack'. It is a story of a 
  // villain --- a performance villain! A villain who's plan was so simple and, dare I say, 
  // dumb that it confounded our heros best efforts to defeat. The villain hid his methods
  // behind C++ copy constructors and the `auto` keyword, and made the heroes chase their
  // tails (which they, unexplainably, grew during this time). Like all good stories involving
  // heros and villains, however, the villain's plot ultimately failed. Our heros found the
  // bug, their tails vanished as mysteriously as they had appeared, and performance was
  // returned to the lands. A mostly happy ending. Mostly, I say, because their victory was 
  // not without some collateral damage. You see, the heros' trek through the lands resulted
  // in a little bit of mayhem. Over the eons of time most of this damage healed, but some
  // scars persisted. 
  //
  // The code below replaced a hash table implementation that performs perfectly well, and 
  // was not a performance bottleneck. We apparently did not undo this hack, however, that
  // was intended to be a temporary performance-debugging step.
  //
  // tldr: Sorry about this hack, the previously used hash table was not a bottleneck. I discovered
  // this was still here during a code audit. If you are reading this, it means we didn't undo it yet. 
  // Could I have undone it in the time it took me to write the story above? Well, let's just say I
  // can type faster than I can think...
  this->fast_cached_vertices_map = new int32_t[200000000];
  this->fast_cached_vertices_isinmap = new bool[200000000]();
  for (int64_t i = 0; i < this->cached_vertices.numel(); i++) {
    this->cached_vertices_map[cached_vertices_data[i]] = i;
    this->fast_cached_vertices_map[cached_vertices_data[i]] = i;
    this->fast_cached_vertices_isinmap[cached_vertices_data[i]] = true;
  }
}

torch::Tensor Cache::nid_is_cached(torch::Tensor nids) const {
  startTime()
  nextTime("placeholder")
  nextTime("a")
  auto nids_data = nids.data_ptr<int64_t>();
  nextTime("b")
  // As far as I know torch::kBool is a byte under the hood.
  // C++ bool type is also a byte.
  torch::Tensor out = torch::empty(nids.numel(), torch::TensorOptions().dtype(torch::kBool));
  nextTime("c")
  //std::cout << "sizeof(torch::kBool): " << sizeof(torch::kBool) << std::endl;
  //std::cout << "sizeof(bool): " << sizeof(bool) << std::endl;
  auto out_data = out.data_ptr<bool>();
  nextTime("d")
  //printf("number of nids %lld\n", nids.numel());
  nextTime("e")
  for (int64_t i = 0; i < nids.numel(); i++) {
    //out_data[i] = this->cached_vertices_map.contains(nids_data[i]); 
    out_data[i] = this->fast_cached_vertices_isinmap[nids_data[i]]; 
  }
  nextTime("f")
  return out;
}

torch::Tensor Cache::nid2cachenid(torch::Tensor nids) const {
  // Assuming that the nids passed in are known to be in the cache.
  auto nids_data = nids.data_ptr<int64_t>();
  torch::Tensor out = torch::empty(nids.numel(), torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  auto out_data = out.data_ptr<int64_t>();
  for (int64_t i = 0; i < nids.numel(); i++) {
    out_data[i] = this->fast_cached_vertices_map[nids_data[i]];
    //out_data[i] = this->cached_vertices_map[nids_data[i]]; 
  }
  return out;
}

