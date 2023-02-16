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

#include "local_cache.h"
#include <algorithm>
#include "rapidjson/document.h"
#include "triton/common/logging.h"

#ifdef TRITON_ENABLE_METRICS
constexpr bool metrics_enabled = true;
#else
constexpr bool metrics_enabled = false;
#endif

namespace helpers {

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

uint64_t
CaptureTimeUs()
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class ScopedTimer {
 public:
  explicit ScopedTimer(uint64_t& duration) : duration_(duration)
  {
    start_ = CaptureTimeUs();
  }

  ~ScopedTimer()
  {
    end_ = CaptureTimeUs();
    duration_ += (end_ - start_);
  }

 private:
  uint64_t start_ = 0;
  uint64_t end_ = 0;
  // Reference passed to update existing variable
  uint64_t& duration_;
};

}  // namespace helpers

namespace triton { namespace cache { namespace local {

TRITONSERVER_Error*
CopyAttributes(
    TRITONSERVER_BufferAttributes* in, TRITONSERVER_BufferAttributes* out)
{
  size_t byte_size = 0;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  // Get attrs
  RETURN_IF_ERROR(TRITONSERVER_BufferAttributesByteSize(in, &byte_size));
  RETURN_IF_ERROR(TRITONSERVER_BufferAttributesMemoryType(in, &memory_type));
  RETURN_IF_ERROR(
      TRITONSERVER_BufferAttributesMemoryTypeId(in, &memory_type_id));
  // Set attrs
  RETURN_IF_ERROR(TRITONSERVER_BufferAttributesSetByteSize(out, byte_size));
  RETURN_IF_ERROR(TRITONSERVER_BufferAttributesSetMemoryType(out, memory_type));
  RETURN_IF_ERROR(
      TRITONSERVER_BufferAttributesSetMemoryTypeId(out, memory_type_id));
  return nullptr;  // success
}

/* LocalCache Implementation */
LocalCache::LocalCache(uint64_t size)
{
  // Allocate buffer
  buffer_ = malloc(size);
  // Exit early if buffer allocation failed
  if (!buffer_) {
    throw std::runtime_error("failed to allocate buffer");
  }

  // Create cache as managed buffer
  managed_buffer_ = boost::interprocess::managed_external_buffer(
      boost::interprocess::create_only_t{}, buffer_, size);

  LOG_INFO << "Response Cache is created at '"
           << helpers::PointerToString(buffer_) << "' with size " << size;

  // Metrics
  if (metrics_enabled) {
    auto err = InitMetrics();
    if (err != nullptr) {
      LOG_ERROR << "Failed to initialize metrics: "
                << TRITONSERVER_ErrorMessage(err);
    }
  }

  return;
}

LocalCache::~LocalCache()
{
  LOG_INFO << "Cleaning up LocalCache";
  // Signal metrics thread to exit and wait for it
  if (metrics_thread_ != nullptr) {
    metrics_thread_exit_.store(true);
    metrics_thread_->join();
  }

  // Deallocate each chunk from managed buffer
  for (auto& iter : cache_) {
    auto& entry = iter.second;
    for (auto& item : entry.items_) {
      for (auto& [buffer, byte_size] : item.buffers_) {
        if (buffer != nullptr) {
          // TODO
          managed_buffer_.deallocate(buffer);
        }
      }
    }
  }

  // Validate we freed all underlying memory managed by cache
  if (!managed_buffer_.all_memory_deallocated()) {
    // Destructors can't throw exceptions
    LOG_ERROR << "failed to free managed cache memory";
  }

  // Free total cache buffer
  if (buffer_ != nullptr) {
    free(buffer_);
  }
}

TRITONSERVER_Error*
LocalCache::Allocate(uint64_t byte_size, void** buffer)
{
  // NOTE: Could have more fine-grained locking, or remove Evict()
  //       from this function and call separately
  std::unique_lock lk(buffer_mu_);

  // Requested buffer larger than total buffer
  if (byte_size > managed_buffer_.get_size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "Requested byte_size: " + std::to_string(byte_size) +
            " is greater than total cache size: " +
            std::to_string(managed_buffer_.get_size()))
            .c_str());
  }
  // Attempt to allocate buffer from current available space
  void* lbuffer = nullptr;
  while (!lbuffer) {
    lbuffer = managed_buffer_.allocate(byte_size, std::nothrow_t{});
    // There wasn't enough available space, so evict and try again
    if (!lbuffer) {
      // Fail if we run out of things to evict
      RETURN_IF_ERROR(Evict());
    }
  }
  // Return allocated buffer
  *buffer = lbuffer;
  return nullptr;  // success
}

TRITONSERVER_Error*
LocalCache::Create(
    const std::string& cache_config, std::unique_ptr<LocalCache>* cache)
{
  LOG_INFO << "Initializing LocalCache with config: " << cache_config;
  rapidjson::Document document;
  document.Parse(cache_config.c_str());
  if (!document.HasMember("size")) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Failed to initialize LocalCache: config didn't contain a valid 'size' "
        "field.");
  }

  try {
    // Parse size field and validate typing
    const auto& size_json = document["size"];
    uint64_t cache_size = 0;
    if (size_json.IsString()) {
      auto size_str = size_json.GetString();
      cache_size = std::stoull(size_str);
    } else if (size_json.IsUint()) {
      cache_size = size_json.GetUint();
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Failed to initialize LocalCache: 'size' config must be a string or "
          "uint type");
    }

    cache->reset(new LocalCache(cache_size));
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "Failed to initialize LocalCache: " + std::string(ex.what()))
            .c_str());
  }

  return nullptr;  // success
}

bool
LocalCache::Exists(const std::string& key)
{
  std::unique_lock lk(cache_mu_);
  return cache_.find(key) != cache_.end();
}

TRITONSERVER_Error*
LocalCache::Lookup(
    const std::string& key, TRITONCACHE_CacheEntry* triton_entry,
    TRITONCACHE_Allocator* allocator)
{
  std::unique_lock lk(cache_mu_);

  // total_lookup_latency must be protected by mutex
  helpers::ScopedTimer timer(total_lookup_latency_us_);
  num_lookups_++;  // must be protected

  auto iter = cache_.find(key);
  if (iter == cache_.end()) {
    num_misses_++;  // must be protected
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("key [" + key + "] does not exist").c_str());
    return err;
  }

  num_hits_++;  // must be protected
  UpdateLRU(iter);

  // Build TRITONCACHE_CacheEntry from cache representation of entry
  auto entry = iter->second;
  for (auto& item : entry.items_) {
    TRITONCACHE_CacheEntryItem* triton_item = nullptr;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemNew(&triton_item));
    for (const auto [buffer, attrs] : item.buffers_) {
      // TODO

      // Create copy of buffer attrs to return to Triton
      TRITONSERVER_BufferAttributes* attrs_copy = nullptr;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesNew(&attrs_copy));
      RETURN_IF_ERROR(CopyAttributes(attrs, attrs_copy));

      size_t byte_size = 0;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size));
      if (!buffer || !attrs || !byte_size) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "buffer or attrs was null, or size was zero");
      }

      // Allocator callback will be used to copy all entry buffers in Triton
      // before this function returns to avoid pre-mature eviction

      // TODO
      RETURN_IF_ERROR(
          TRITONCACHE_CacheEntryItemAddBuffer(triton_item, buffer, attrs_copy));
    }

    // Pass ownership of triton_item to Triton to avoid copy. Triton will
    // be responsible for cleaning it up, so do not call CacheEntryItemDelete.

    RETURN_IF_ERROR(TRITONCACHE_CacheEntryAddItem(triton_entry, triton_item));
  }

  // Copy entry/item buffers directly into allocator-provided buffers
  return TRITONCACHE_Copy(allocator, triton_entry);
}

TRITONSERVER_Error*
LocalCache::Insert(
    const std::string& key, CacheEntry& entry, TRITONCACHE_Allocator* allocator)
{
  std::unique_lock lk(cache_mu_);
  // total_insertion_latency must be protected by mutex
  helpers::ScopedTimer timer(total_insertion_latency_us_);

  // Check that sum of entry sizes does not exceed total available cache size
  {
    std::unique_lock lk(buffer_mu_);
    uint64_t total_entry_size = 0;
    for (auto& item : entry.items_) {
      for (auto& [base, attrs] : item.buffers_) {
        size_t byte_size = 0;
        RETURN_IF_ERROR(
            TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size));
        total_entry_size += byte_size;
      }
    }

    if (total_entry_size > managed_buffer_.get_size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "Requested byte_size: " + std::to_string(total_entry_size) +
              " is greater than total cache size: " +
              std::to_string(managed_buffer_.get_size()))
              .c_str());
    }
  }

  // Exit early if key already exists to avoid setting up entry unnecessarily
  if (cache_.find(key) != cache_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        std::string("Insert failed, key [" + key + "] already exists in cache")
            .c_str());
  }

  // Allocate and copy into a chunk from managed buffer for each item in entry
  // NOTE: probably a cleaner way to do this
  for (auto& item : entry.items_) {
    for (size_t b = 0; b < item.buffers_.size(); b++) {
      auto& [base, attrs] = item.buffers_[b];
      // TODO

      size_t byte_size = 0;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size));
      // Copy triton contents into cache representation for cache to own
      void* new_base = nullptr;
      // Request block of memory from cache
      RETURN_IF_ERROR(Allocate(byte_size, &new_base));
      // TODO
      RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemSetBuffer(
          item.triton_item_, b, new_base, attrs));
      base = new_base;
    }
  }

  // Let Triton copy directly into cache buffers to avoid intermediate copies
  RETURN_IF_ERROR(TRITONCACHE_Copy(allocator, entry.triton_entry_));

  auto [iter, success] = cache_.insert({key, entry});
  if (!success) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        std::string("Failed to insert key [" + key + "] into map").c_str());
  }
  UpdateLRU(iter);
  return nullptr;  // success
}

TRITONSERVER_Error*
LocalCache::Evict()
{
  // NOTE: Unique lock on cache and buffer mutexes must be held for this
  // function
  // Nothing to evict if cache is empty
  if (cache_.size() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "Cache is empty, nothing to evict.");
  }

  // Least recently used key in back of LRU list
  auto lru_key = lru_.back();
  LOG_VERBOSE(1) << "Evicting key [" + lru_key + "] from cache.";

  // Find cache entry for least recently used key
  auto iter = cache_.find(lru_key);
  if (iter == cache_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "key [" + lru_key +
            "] not found in cache during eviction: this likely indicates a bug")
            .c_str());
  }

  // Get size of cache entry being evicted to update available size
  auto entry = iter->second;
  // Free managed memory used in cache entry's outputs
  for (auto& item : entry.items_) {
    for (auto& [base, byte_size] : item.buffers_) {
      // Lock on managed_buffer assumed to be already held
      managed_buffer_.deallocate(base);
    }
  }

  // Remove LRU entry from cache
  cache_.erase(lru_key);
  // Remove LRU key from LRU list
  lru_.pop_back();

  num_evictions_++;  // must be protected
  return nullptr;    // success
}

// Helpers
void
LocalCache::UpdateLRU(
    std::unordered_map<std::string, CacheEntry>::iterator& cache_iter)
{
  // NOTE: Unique lock on cache mutex must be held for this function

  const auto& key = cache_iter->first;
  auto& cache_entry = cache_iter->second;
  // Remove key from LRU list if it was already in there
  auto lru_iter = std::find(lru_.begin(), lru_.end(), key);
  if (lru_iter != lru_.end()) {
    lru_.erase(lru_iter);
  }
  // Add key to front of LRU list since it's most recently used
  lru_.push_front(key);
  // Set CacheEntry LRU iterator to new LRU key location
  cache_entry.lru_iter_ = lru_.begin();
}


// Cache Metric Helpers: these must be protected when accessed
TRITONSERVER_Error*
LocalCache::InitMetrics()
{
  if (!metrics_enabled) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
  }

  // Initialize Triton cache metrics
  const TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_GAUGE;
  RETURN_IF_ERROR(
      cache_util_.Init(kind, "nv_cache_util", "Cache utilization [0.0 - 1.0]"));
  RETURN_IF_ERROR(cache_entries_.Init(
      kind, "nv_cache_num_entries",
      "Number of responses stored in response cache"));
  RETURN_IF_ERROR(cache_hits_.Init(
      kind, "nv_cache_num_hits", "Number of cache hits in response cache"));
  RETURN_IF_ERROR(cache_misses_.Init(
      kind, "nv_cache_num_misses", "Number of cache misses in response cache"));
  RETURN_IF_ERROR(cache_lookups_.Init(
      kind, "nv_cache_num_lookups",
      "Number of cache lookups in response cache"));
  RETURN_IF_ERROR(cache_evictions_.Init(
      kind, "nv_cache_num_evictions",
      "Number of cache evictions in response cache"));
  RETURN_IF_ERROR(cache_lookup_time_.Init(
      kind, "nv_cache_lookup_duration",
      "Total cache lookup duration (hit and miss), in microseconds"));
  RETURN_IF_ERROR(cache_insert_time_.Init(
      kind, "nv_cache_insertion_duration",
      "Total cache insertion duration, in microseconds"));

  // Start thread that polls metrics at an interval
  metrics_thread_exit_.store(false);
  metrics_thread_.reset(new std::thread([this] {
    while (!metrics_thread_exit_.load()) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(metrics_interval_ms_));
      // An error is not expected here, but log the error if it occurs
      auto err = UpdateMetrics();
      if (err != nullptr) {
        LOG_ERROR << TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }));

  return nullptr;  // success
}


TRITONSERVER_Error*
LocalCache::UpdateMetrics()
{
  if (!metrics_enabled) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
  }

  // Query and update cache metrics
  {
    std::unique_lock clk(cache_mu_);
    {
      std::unique_lock blk(buffer_mu_);
      RETURN_IF_ERROR(cache_util_.Set(TotalUtilization()));
    }
    RETURN_IF_ERROR(cache_entries_.Set(NumEntries()));
    RETURN_IF_ERROR(cache_hits_.Set(num_hits_));
    RETURN_IF_ERROR(cache_misses_.Set(num_misses_));
    RETURN_IF_ERROR(cache_lookups_.Set(num_lookups_));
    RETURN_IF_ERROR(cache_evictions_.Set(num_evictions_));
    RETURN_IF_ERROR(cache_lookup_time_.Set(total_lookup_latency_us_));
    RETURN_IF_ERROR(cache_insert_time_.Set(total_insertion_latency_us_));
  }

  return nullptr;  // success
}

uint64_t
LocalCache::NumEntries()
{
  // Must hold buffer_mu_ to safely access this
  return cache_.size();
}

// Returns fraction of bytes allocated over total cache size between [0, 1]
double
LocalCache::TotalUtilization()
{
  // Must hold buffer_mu_ to safely access this
  const auto total = managed_buffer_.get_size();
  const auto used = total - managed_buffer_.get_free_memory();
  return static_cast<double>(used) / static_cast<double>(total);
}

}}}  // namespace triton::cache::local
