#include "local_cache.h"
#include <iostream>  // TODO

namespace triton { namespace cache { namespace local {


/* LocalCache Implementation */
LocalCache::LocalCache(uint64_t size)
{
  // TODO
  std::cout << "LocalCache::LocalCache() constructor called" << std::endl;
  return;
}

LocalCache::~LocalCache()
{
  // TODO
  std::cout << "LocalCache::~LocalCache() destructor called" << std::endl;
}

TRITONSERVER_Error*
LocalCache::Create(uint64_t cache_size, std::unique_ptr<LocalCache>* cache)
{
  std::cout << "LocalCache::Create() called" << std::endl;
  cache->reset(new LocalCache(cache_size));
  return nullptr;  // success
}

bool
LocalCache::Exists(const std::string& key)
{
  // TODO: Can we allocate a map (ease of use) with a fixed size pool managed by
  // boost?

  // TODO: read-only lock on map
  return map_.find(key) != map_.end();
}

std::pair<TRITONSERVER_Error*, CacheEntry>
LocalCache::Lookup(const std::string& key)
{
  // TODO
  std::cout << "[DEBUG] [local_cache.cc] LocalCache::Lookup() with key: " << key
            << std::endl;
  // TODO: read-only lock
  const auto iter = map_.find(key);
  if (iter == map_.end()) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("key [" + key + "] does not exist").c_str());
    return {err, {}};
  }
  const auto entry = iter->second;
  // TODO: Remove
  std::cout << "[DEBUG] [local_cache.cc] LocalCache::Lookup() FOUND key: "
            << key << std::endl;
  std::cout
      << "[DEBUG] [local_cache.cc] LocalCache::Lookup entry.items_.size(): "
      << entry.items_.size() << std::endl;
  // TODO: Use std::optional instead
  return std::make_pair(nullptr, entry);  // success
}

TRITONSERVER_Error*
LocalCache::Insert(const std::string& key, const CacheEntry& entry)
{
  // TODO
  std::cout << "[DEBUG] [local_cache.cc] LocalCache::Insert() with key: " << key
            << std::endl;
  // TODO: read+write lock
  if (map_.find(key) != map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("key [" + key + "] already exists").c_str());
  }

  // TODO: Remove
  auto litems = entry.items_;
  for (const auto& buffer : litems) {
    std::cout << "[DEBUG] [local_cache.cc] [INSERT] buffer.size(): "
              << buffer.size() << std::endl;
    if (!buffer.size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
    }
  }

  map_[key] = entry;
  return nullptr;  // success
}

std::pair<TRITONSERVER_Error*, CacheEntry>
LocalCache::EntryFromTriton(TRITONCACHE_CacheEntry*)
{
  CacheEntry lentry;
  // TODO: Create internal entry from opaque triton entry through C APIs
  // items = TRITONCACHE_CacheEntryItems(...), tags =
  // TRITONCACHE_CacheEntryTags(...)
  return std::make_pair(nullptr, lentry);  // success
}

std::pair<TRITONSERVER_Error*, TRITONCACHE_CacheEntry*>
LocalCache::EntryToTriton(const CacheEntry& entry)
{
  TRITONCACHE_CacheEntry* lentry = nullptr;
  // TODO: Setup triton entry, TritonCacheEntryNew, SetItems, SetTags
  // TODO: Create internal entry from opaque triton entry through C APIs
  // items = TRITONCACHE_CacheEntryItems(...), tags =
  // TRITONCACHE_CacheEntryTags(...)
  return std::make_pair(nullptr, lentry);  // success
}


}}}  // namespace triton::cache::local
