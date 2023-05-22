#ifndef RANGE_PARTITION_BOOK_H
#define RANGE_PARTITION_BOOK_H



#include <string>
#include <pybind11/chrono.h>
#include <semaphore.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

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

#include "parallel_hashmap/phmap.h"


class RangePartitionBook {
  public:

    RangePartitionBook(){};

    RangePartitionBook(
      int rank_,
      int world_size_,
      torch::Tensor partition_offsets_
    );

    int world_size;
    int rank;
    torch::Tensor partition_offsets; 

    // Convert global nids to local ids within the partition.
    torch::Tensor nid2localnid(torch::Tensor nids, int partition_idx) const;

    // Map a tensor of global nids to a tensor of partition ids which they belong to.
    torch::Tensor nid2partid(torch::Tensor nids) const;

    // Return a boolean tensor of whether the nids are in the local parition.
    torch::Tensor nid_is_local(torch::Tensor nids) const;

    // Given a partition idx return all global nids local to that partition.
    torch::Tensor partid2nids(int partition_idx) const;
};


class Cache {
  public:

    Cache();

    Cache(
      int rank_,
      int world_size_,
      torch::Tensor cached_vertices_,
      torch::Tensor cached_features_
    );

    int rank;
    int world_size;
    torch::Tensor cached_vertices;
    torch::Tensor cached_features;
    int32_t* fast_cached_vertices_map;
    bool* fast_cached_vertices_isinmap;


    // Return a bool tensor of whether the nids are cached.
    torch::Tensor nid_is_cached(torch::Tensor nids) const;

    // Convert global nids to local ids within the the cache.
    // Not important but a little suspicious that had to remove const keyword to index into cached_vertices_map.
    torch::Tensor nid2cachenid(torch::Tensor nids) const;

  private:
    // Internal map, profile if a bottlenecks but should be a fast way to check if things are in cache.
    // keys are global indices, values are local indices. 
    phmap::flat_hash_map<int64_t, int64_t> cached_vertices_map;
};



#endif // RANGE_PARTITION_BOOK_H
