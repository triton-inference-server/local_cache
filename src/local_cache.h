// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <atomic>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

#define RETURN_IF_ERROR(X)        \
  do {                            \
    TRITONSERVER_Error* err__(X); \
    if (err__ != nullptr) {       \
      return err__;               \
    }                             \
  } while (false)

namespace triton { namespace cache { namespace local {

using Buffer = std::pair<void*, TRITONSERVER_BufferAttributes*>;

TRITONSERVER_Error* CopyAttributes(
    TRITONSERVER_BufferAttributes* in, TRITONSERVER_BufferAttributes* out);

struct CacheEntryItem {
  std::vector<Buffer> buffers_;
  TRITONCACHE_CacheEntryItem* triton_item_;
};

struct CacheEntry {
  std::vector<CacheEntryItem> items_;
  // Point to key in LRU list for maintaining LRU order
  std::list<std::string>::iterator lru_iter_;
  TRITONCACHE_CacheEntry* triton_entry_;
};

struct TritonMetric {
  TRITONSERVER_MetricFamily* family_ = nullptr;
  TRITONSERVER_Metric* metric_ = nullptr;

  TRITONSERVER_Error* Init(
      TRITONSERVER_MetricKind kind, const char* name, const char* description)
  {
    RETURN_IF_ERROR(
        TRITONSERVER_MetricFamilyNew(&family_, kind, name, description));
    std::vector<const TRITONSERVER_Parameter*> labels;
    RETURN_IF_ERROR(TRITONSERVER_MetricNew(
        &metric_, family_, labels.data(), labels.size()));
    return nullptr;  // success
  }

  TRITONSERVER_Error* Set(double value)
  {
    if (!metric_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "metric was nullptr");
    }
    RETURN_IF_ERROR(TRITONSERVER_MetricSet(metric_, value));
    return nullptr;  // success
  }

  ~TritonMetric()
  {
    if (metric_) {
      TRITONSERVER_MetricDelete(metric_);
    }
    if (family_) {
      TRITONSERVER_MetricFamilyDelete(family_);
    }
  }
};

class LocalCache {
 public:
  ~LocalCache();

  // Create the cache object
  static TRITONSERVER_Error* Create(
      const std::string& cache_config, std::unique_ptr<LocalCache>* cache);

  // Lookup key in cache and return the data associated with it
  // Return TRITONSERVER_Error* object indicating success or failure.
  // std::pair<TRITONSERVER_Error*, CacheEntry> Lookup(
  TRITONSERVER_Error* Lookup(
      const std::string& key, TRITONCACHE_CacheEntry* entry,
      TRITONCACHE_Allocator* allocator);

  // Insert entry into cache, evict entries to make space if necessary
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Insert(
      const std::string& key, CacheEntry& entry,
      TRITONCACHE_Allocator* allocator);

  // Checks if key exists in cache
  // Return true if key exists in cache, false otherwise.
  bool Exists(const std::string& key);

  // Evict entries from cache based on policy.
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Evict();

 private:
  LocalCache(uint64_t size);
  // Update ordering used for LRU eviction policy
  void UpdateLRU(
      std::unordered_map<std::string, CacheEntry>::iterator& cache_iter);
  // Request buffer from managed buffer
  TRITONSERVER_Error* Allocate(uint64_t byte_size, void** buffer);

  // Cache Metric Helpers: note the metrics must be protected when accessed
  TRITONSERVER_Error* InitMetrics();
  TRITONSERVER_Error* UpdateMetrics();
  // Returns the number of entries currently in cache
  uint64_t NumEntries();
  // Returns fraction of bytes allocated over total cache size between [0, 1]
  double TotalUtilization();

  // Cache Metrics: these must be protected when accessed
  uint64_t num_evictions_ = 0;
  uint64_t num_lookups_ = 0;
  uint64_t num_hits_ = 0;
  uint64_t num_misses_ = 0;
  // Lookup/Insert latencies in microseconds
  uint64_t total_lookup_latency_us_ = 0;
  uint64_t total_insertion_latency_us_ = 0;
  // TRITONSERVER_Metric/Family wrappers
  TritonMetric cache_util_;
  TritonMetric cache_entries_;
  TritonMetric cache_hits_;
  TritonMetric cache_misses_;
  TritonMetric cache_lookups_;
  TritonMetric cache_evictions_;
  TritonMetric cache_lookup_time_;
  TritonMetric cache_insert_time_;
  // Thread to poll metrics at an interval
  std::unique_ptr<std::thread> metrics_thread_;
  std::atomic<bool> metrics_thread_exit_ = false;
  // The interval at which metrics are updated by metric thread.
  // Default interval is 1000ms = 1sec.
  // NOTE: Can expose this as config field in the future
  uint64_t metrics_interval_ms_ = 1000;

  // Underlying contiguous buffer managed by managed_buffer_
  void* buffer_;
  // Managed buffer
  boost::interprocess::managed_external_buffer managed_buffer_;
  // Protect concurrent cache access
  std::shared_mutex cache_mu_;
  // Protect concurrent managed buffer access
  std::mutex buffer_mu_;
  // key -> CacheEntry containing values and list iterator for LRU management
  std::unordered_map<std::string, CacheEntry> cache_;
  // List of keys sorted from most to least recently used
  std::list<std::string> lru_;
};

}}}  // namespace triton::cache::local
