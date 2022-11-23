#include <boost/core/span.hpp>  // TODO: remove
#include <iostream>             // TODO: remove
#include "local_cache.h"
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

#define RETURN_IF_ERROR(X)           \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)

// TODO: Remove
void
printBytes(boost::span<const std::byte> buffer)
{
  // Capture blank std::cout state
  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  std::cout << "[DEBUG] [cache_api.cc] [LOOKUP] Buffer bytes: ";
  for (const auto& byte : buffer) {
    std::cout << std::hex << "0x" << std::to_integer<int>(byte) << " ";
  }
  std::cout << std::endl;

  // Reset std::cout state
  std::cout.copyfmt(oldState);
}

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
  for (const auto& item : lentry.items_) {
    TRITONCACHE_CacheEntryItem* triton_item = nullptr;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemNew(&triton_item));
    // TODO: buffer size == 10 and not 16 ?
    for (const auto& buffer : item.buffers_) {
      std::cout << "[DEBUG] [cache_api.cc] [LOOKUP] buffer.size(): "
                << buffer.size() << std::endl;
      if (!buffer.size()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
      }

      // TODO: Remove
      printBytes(buffer);

      RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemAddBuffer(
          triton_item, buffer.data(), buffer.size()));
    }

    RETURN_IF_ERROR(TRITONCACHE_CacheEntryAddItem(entry, triton_item));
    RETURN_IF_ERROR(
        TRITONCACHE_CacheEntryItemDelete(triton_item));  // TODO: check valid
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
  RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemCount(entry, &num_items));
  std::cout << "[DEBUG] [cache_api.cc] [INSERT] num_items " << num_items
            << std::endl;

  // Form cache representation of CacheEntry from Triton
  CacheEntry lentry;
  for (size_t item_index = 0; item_index < num_items; item_index++) {
    TRITONCACHE_CacheEntryItem* item = nullptr;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryGetItem(entry, item_index, &item));

    size_t num_buffers = 0;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemBufferCount(item, &num_buffers));

    // Form cache representation of CacheEntryItem from Triton
    CacheEntryItem litem;
    for (size_t buffer_index = 0; buffer_index < num_buffers; buffer_index++) {
      void* base = nullptr;
      size_t byte_size = 0;
      RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemGetBuffer(
          item, buffer_index, &base, &byte_size));
      std::cout << "[DEBUG] [cache_api.cc] [INSERT] byte_size " << byte_size
                << std::endl;
      // TODO
      if (!byte_size) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
      }

      auto byte_base = reinterpret_cast<std::byte*>(base);

      // Copy triton contents into cache representation for cache to own
      litem.buffers_.emplace_back(byte_base, byte_base + byte_size);
    }
    lentry.items_.emplace_back(litem);
  }

  const auto lcache = reinterpret_cast<LocalCache*>(cache);
  lcache->Insert(key, lentry);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::cache::local
