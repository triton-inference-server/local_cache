#include <memory>
#include <vector>
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace cache { namespace local {

struct CacheEntry {
  void* data;
  size_t size;
};

class LocalCache {
 public:
  ~LocalCache();

  // Create the cache object
  static TRITONSERVER_Error* Create(
      uint64_t cache_size, std::unique_ptr<LocalCache>* cache);

  // Lookup key in cache and return the data associated with it
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Lookup(std::string key, std::vector<CacheEntry>* entries);

  // Insert entries into cache, evict entries to make space if necessary
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Insert(
      std::string key, const std::vector<CacheEntry>& entries);

  // Evict entries from cache based on policy.
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Evict();

 private:
  // TODO
  LocalCache(uint64_t size);
};

}}}  // namespace triton::cache::local
