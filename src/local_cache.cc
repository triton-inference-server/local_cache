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

}}}  // namespace triton::cache::local
