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
  // Read-only, can be shared
  std::shared_lock lk(map_mu_);
  return map_.find(key) != map_.end();
}

std::pair<TRITONSERVER_Error*, CacheEntry>
LocalCache::Lookup(const std::string& key)
{
  // Read-only, can be shared
  std::shared_lock lk(map_mu_);

  const auto iter = map_.find(key);
  if (iter == map_.end()) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("key [" + key + "] does not exist").c_str());
    return {err, {}};
  }
  return std::make_pair(nullptr, iter->second);  // success
}

TRITONSERVER_Error*
LocalCache::Insert(const std::string& key, const CacheEntry& entry)
{
  // Read-write, cannot be shared
  std::unique_lock lk(map_mu_);

  if (map_.find(key) != map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        std::string("key [" + key + "] already exists").c_str());
  }

  map_[key] = entry;
  return nullptr;  // success
}

}}}  // namespace triton::cache::local
