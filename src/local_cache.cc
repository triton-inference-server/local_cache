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

namespace helpers {
std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}
}  // namespace helpers

namespace triton { namespace cache { namespace local {


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
  return;
}

LocalCache::~LocalCache()
{
  // Deallocate each chunk from managed buffer
  for (auto& iter : cache_) {
    auto& entry = iter.second;
    for (auto& item : entry.items_) {
      for (auto& [buffer, byte_size] : item.buffers_) {
        if (buffer != nullptr) {
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
  // TODO: more fine-grained locking, or remove Evict()
  //       from this function and call separately
  std::unique_lock lk(buffer_mu_);
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
  rapidjson::Document document;
  document.Parse(cache_config.c_str());
  if (!document.HasMember("size") || !document["size"].IsUint()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Failed to initialize LocalCache: config didn't contain a valid 'size' "
        "field.");
  }
  uint64_t cache_size = document["size"].GetUint64();

  try {
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

std::pair<TRITONSERVER_Error*, CacheEntry>
LocalCache::Lookup(const std::string& key)
{
  std::unique_lock lk(cache_mu_);

  auto iter = cache_.find(key);
  if (iter == cache_.end()) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("key [" + key + "] does not exist").c_str());
    return {err, {}};
  }
  UpdateLRU(iter);
  return std::make_pair(nullptr, iter->second);  // success
}

TRITONSERVER_Error*
LocalCache::Insert(const std::string& key, CacheEntry& entry)
{
  std::unique_lock lk(cache_mu_);

  // TODO: probably a cleaner way to do this
  for (auto& item : entry.items_) {
    for (auto& [base, byte_size] : item.buffers_) {
      // Copy triton contents into cache representation for cache to own
      void* new_base = nullptr;
      // Request block of memory from cache
      RETURN_IF_ERROR(Allocate(byte_size, &new_base));
      // Copy contents into cache-allocated buffer
      std::memcpy(new_base, base, byte_size);
      // Replace triton pointer with cache-allocated pointer in cache entry
      base = new_base;
    }
  }

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

  return nullptr;  // success
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

}}}  // namespace triton::cache::local
