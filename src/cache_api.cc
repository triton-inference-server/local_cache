#include "local_cache.h"

namespace triton { namespace cache { namespace local {

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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }

  delete reinterpret_cast<LocalCache*>(cache);
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_CacheLookup(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry)
{
  // TODO
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  } else if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache entry was nullptr");
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
    lsizes.emplace_back(entry.size);  // TODO
  }
  // TODO: fix lifetimes here
  auto num_items = lentries.size();
  auto byte_sizes = lsizes.data();
  auto items = ldata.data();
  // TODO: Need to create entry here, or creating/passing entry from Triton is
  // OK?
  err = TRITONCACHE_CacheEntrySetItems(entry, items, byte_sizes, num_items);
  if (err != nullptr) {
    return err;
  }

  // TODO: Set tags if used at all too
  // err = TRITONCACHE_CacheEntrySetTags(...)

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::cache::local
