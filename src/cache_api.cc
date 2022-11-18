#include <boost/core/span.hpp>  // TODO: remove
#include <iostream>             // TODO: remove
#include "local_cache.h"
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace cache { namespace local {

extern "C" {

TRITONSERVER_Error*
TRITONCACHE_CacheNew(
    TRITONCACHE_Cache** cache, TRITONSERVER_Message* cache_config)
{
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }
  // TODO
  if (cache_config == nullptr) {
    std::cout
        << "[DEBUG] [cache_api.cc] [TRITONCACHE_CacheNew] cache_config NOT "
           "IMPLEMENTED YET"
        << std::endl;
  }

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

  const auto lcache = reinterpret_cast<LocalCache*>(cache);
  auto [err, lentry] = lcache->Lookup(key);
  if (err != nullptr) {
    return err;
  }

  // TODO - cache entry class like core instead of struct?
  std::cout
      << "[DEBUG] [cache_api.cc] Building TRITONCACHE_CacheEntry to return"
      << std::endl;
  std::cout << "[DEBUG] [cache_api.cc] lentry.items_.size(): "
            << lentry.items_.size() << std::endl;
  auto litems = lentry.items_;
  for (const auto& buffer : litems) {
    std::cout << "[DEBUG] [cache_api.cc] [LOOKUP] buffer.size(): "
              << buffer.size() << std::endl;
    if (!buffer.size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
    }

    // TODO: Remove
    std::cout << "[DEBUG] [cache_api.cc] [LOOKUP] entry ints: ";
    int* ibuffer = (int*)(buffer.data());
    for (size_t i = 0; i < (buffer.size() / sizeof(int)); i++) {
      std::cout << ibuffer[i] << " ";
    }
    std::cout << std::endl;
    // TODO: Use RETURN_IF_ERROR
    err = TRITONCACHE_CacheEntryAddItem(entry, buffer.data(), buffer.size());
    if (err != nullptr) {
      return err;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_CacheInsert(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry)
{
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  } else if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache entry was nullptr");
  } else if (key == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "key was nullptr");
  }

  size_t num_items = 0;
  auto err = TRITONCACHE_CacheEntryItemCount(entry, &num_items);
  if (err != nullptr) {
    return err;
  }
  std::cout << "[DEBUG] [cache_api.cc] [INSERT] num_items " << num_items
            << std::endl;

  CacheEntry lentry;
  for (size_t index = 0; index < num_items; index++) {
    void* base = nullptr;
    size_t byte_size = 0;
    auto err = TRITONCACHE_CacheEntryItem(entry, index, &base, &byte_size);
    if (err != nullptr) {
      return err;
    }
    std::cout << "[DEBUG] [cache_api.cc] [INSERT] byte_size " << byte_size
              << std::endl;
    // TODO
    if (!byte_size) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
    }

    auto byte_base = reinterpret_cast<std::byte*>(base);
    lentry.items_.emplace_back(byte_base, byte_base + byte_size);
  }

  const auto lcache = reinterpret_cast<LocalCache*>(cache);
  lcache->Insert(key, lentry);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::cache::local
