#include <boost/interprocess/managed_external_buffer.hpp>
#include <list>
#include <memory>
#include <shared_mutex>
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
  // Shared mutex to support read-only and read-write locks
  std::shared_mutex cache_mu_;
  std::shared_mutex buffer_mu_;
  // key -> CacheEntry containing values and list iterator for LRU management
  std::unordered_map<std::string, CacheEntry> cache_;
  // List of keys sorted from most to least recently used
  std::list<std::string> lru_;
};

}}}  // namespace triton::cache::local
