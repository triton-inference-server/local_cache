#include <memory>
#include <unordered_map>  // TODO
#include <vector>
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace cache { namespace local {

using Buffer = std::vector<std::byte>;  // raw buffers

struct CacheEntryItem {
  std::vector<Buffer> buffers_;
};

struct CacheEntry {
  std::vector<CacheEntryItem> items_;
};

class LocalCache {
 public:
  ~LocalCache();

  // Create the cache object
  static TRITONSERVER_Error* Create(
      uint64_t cache_size, std::unique_ptr<LocalCache>* cache);

  // Lookup key in cache and return the data associated with it
  // Return TRITONSERVER_Error* object indicating success or failure.
  std::pair<TRITONSERVER_Error*, CacheEntry> Lookup(const std::string& key);

  // Insert entry into cache, evict entries to make space if necessary
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Insert(const std::string& key, const CacheEntry& entry);

  // Checks if key exists in cache
  // Return true if key exists in cache, false otherwise.
  bool Exists(const std::string& key);

  // Evict entries from cache based on policy.
  // Return TRITONSERVER_Error* object indicating success or failure.
  TRITONSERVER_Error* Evict();

 private:
  // TODO
  LocalCache(uint64_t size);

  // Helpers
  std::pair<TRITONSERVER_Error*, TRITONCACHE_CacheEntry*> EntryToTriton(
      const CacheEntry& entry);
  std::pair<TRITONSERVER_Error*, CacheEntry> EntryFromTriton(
      TRITONCACHE_CacheEntry* entry);

  // TODO: map backed by boost buffer?
  std::unordered_map<std::string, CacheEntry> map_;
};

}}}  // namespace triton::cache::local
