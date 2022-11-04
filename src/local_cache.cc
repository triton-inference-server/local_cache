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

TRITONSERVER_Error*
LocalCache::Lookup(std::string key, std::vector<CacheEntry>* entries)
{
  // TODO
  std::cout << "LocalCache::Lookup() with key: " << key << std::endl;
  *entries = {};
  return nullptr;  // success
}

/* C APIs */

// NOTES: (Remove)
//   DECLSPEC for Triton-Core defined methods?
//   ISPEC for Cache Implementation defined methods?

extern "C" {

TRITONSERVER_Error*
TRITONCACHE_CacheNew(
    TRITONCACHE_Cache** cache, TRITONSERVER_Message* cache_config)
{
  // TODO: Parse cache config for size
  std::unique_ptr<LocalCache> lcache;
  constexpr auto cache_size = 4 * 1024 * 1024;  // 4 MB
  LocalCache::Create(cache_size, &lcache);
  *cache = reinterpret_cast<TRITONCACHE_Cache*>(lcache.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_CacheDelete(TRITONCACHE_Cache* cache)
{
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }

  delete reinterpret_cast<LocalCache*>(cache);
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_CacheLookup(TRITONCACHE_Cache* cache, const char* key, void** entries, size_t** sizes, size_t* num_entries)
{
  // TODO
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }

  auto lcache = reinterpret_cast<LocalCache*>(cache);
  std::vector<CacheEntry> lentries;
  std::vector<void*> ldata;
  auto err = lcache->Lookup(key, &lentries);
  if (err != nullptr) {
    return err;
  }

  std::vector<size_t> lsizes;
  for (const auto& entry : lentries) {
    ldata.emplace_back(entry.data);
    lsizes.emplace_back(entry.size); // TODO
  }
  // TODO: lifetimes OK?
  *num_entries = lentries.size();
  *sizes = lsizes.data();
  *entries = ldata.data();

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::cache::local
