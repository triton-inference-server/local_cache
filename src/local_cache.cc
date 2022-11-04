#include <iostream> // TODO
#include "local_cache.h"

namespace triton { namespace cache { namespace local {

/* LocalCache Implementation */
LocalCache::LocalCache(uint64_t size) {
  // TODO
  std::cout << "LocalCache::LocalCache() constructor called" << std::endl;
  return;
}

LocalCache::~LocalCache() {
  // TODO
  std::cout << "LocalCache::~LocalCache() destructor called" << std::endl;
}

TRITONSERVER_Error* LocalCache::Create(uint64_t cache_size, std::unique_ptr<LocalCache>* cache) {
  std::cout << "LocalCache::Create() called" << std::endl;
  cache->reset(new LocalCache(cache_size));
  return nullptr;  // success
}

/* C APIs */

// NOTES: (Remove)
//   DECLSPEC for Triton-Core defined methods?
//   ISPEC for Cache Implementation defined methods?

extern "C" {

TRITONSERVER_Error*
TRITONCACHE_CacheNew(
  TRITONCACHE_Cache** cache, 
  TRITONSERVER_Message* cache_config
)
{
  // TODO: Parse cache config for size
  std::unique_ptr<LocalCache> lcache;
  constexpr auto cache_size = 4*1024*1024; // 4 MB
  LocalCache::Create(cache_size, &lcache);
  *cache = reinterpret_cast<TRITONCACHE_Cache*>(lcache.release());
  return nullptr;  // success
}

}  // extern "C"

}}}  // triton::cache::local
