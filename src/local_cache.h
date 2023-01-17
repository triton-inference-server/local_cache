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

#include <boost/interprocess/managed_external_buffer.hpp>
#include <list>
#include <memory>
#include <mutex>
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

using Buffer = std::pair<void*, size_t>;  // pointer and size

struct CacheEntryItem {
  std::vector<Buffer> buffers_;
};

struct CacheEntry {
  std::vector<CacheEntryItem> items_;
  // Point to key in LRU list for maintaining LRU order
  std::list<std::string>::iterator lru_iter_;
};

class LocalCache {
 public:
  ~LocalCache();

  // Create the cache object
  static TRITONSERVER_Error* Create(
      const std::string& cache_config, std::unique_ptr<LocalCache>* cache);

  // Lookup key in cache and return the data associated with it
  // Return TRITONSERVER_Error* object indicating success or failure.
  std::pair<TRITONSERVER_Error*, CacheEntry> Lookup(const std::string& key);

  // Insert entry into cache, evict entries to make space if necessary
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Insert(const std::string& key, CacheEntry& entry);

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

  // Underlying contiguous buffer managed by managed_buffer_
  void* buffer_;
  // Managed buffer
  boost::interprocess::managed_external_buffer managed_buffer_;
  std::mutex cache_mu_;
  std::mutex buffer_mu_;
  // key -> CacheEntry containing values and list iterator for LRU management
  std::unordered_map<std::string, CacheEntry> cache_;
  // List of keys sorted from most to least recently used
  std::list<std::string> lru_;
};

}}}  // namespace triton::cache::local
