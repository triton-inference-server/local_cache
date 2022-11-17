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
  std::cout << "LocalCache::Lookup() with key: " << key << std::endl;
  // TODO: read-only lock
  const auto iter = map_.find(key);
  if (iter == map_.end()) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("key [" + key + "] does not exist").c_str());
    return {err, {}};
  }
  const auto entry = iter->second;
  std::cout << "LocalCache::Lookup() finished for key: " << key << std::endl;
  return std::make_pair(nullptr, entry);  // success
}

TRITONSERVER_Error*
LocalCache::Insert(const std::string& key, const CacheEntry& entry)
{
  // TODO
  std::cout << "LocalCache::Insert() with key: " << key << std::endl;
  // TODO: read+write lock
  if (map_.find(key) != map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("key [" + key + "] already exists").c_str());
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
