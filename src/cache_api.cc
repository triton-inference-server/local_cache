// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "local_cache.h"
#include "triton/core/tritoncache.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace cache { namespace local {

extern "C" {

TRITONSERVER_Error*
TRITONCACHE_CacheInitialize(TRITONCACHE_Cache** cache, const char* cache_config)
{
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }
  if (cache_config == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  }

  std::unique_ptr<LocalCache> lcache;
  RETURN_IF_ERROR(LocalCache::Create(cache_config, &lcache));
  *cache = reinterpret_cast<TRITONCACHE_Cache*>(lcache.release());
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONCACHE_CacheFinalize(TRITONCACHE_Cache* cache)
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
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry,
    TRITONCACHE_Allocator* allocator)
{
  if (cache == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache was nullptr");
  } else if (entry == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "cache entry was nullptr");
  } else if (allocator == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "allocator was nullptr");
  }

  const auto lcache = reinterpret_cast<LocalCache*>(cache);
  return lcache->Lookup(key, entry, allocator);
}

TRITONSERVER_Error*
TRITONCACHE_CacheInsert(
    TRITONCACHE_Cache* cache, const char* key, TRITONCACHE_CacheEntry* entry,
    TRITONCACHE_Allocator* allocator)
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

  const auto lcache = reinterpret_cast<LocalCache*>(cache);
  if (lcache->Exists(key)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        (std::string("key '") + key + std::string("' already exists")).c_str());
  }

  size_t num_items = 0;
  RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemCount(entry, &num_items));

  // Form cache representation of CacheEntry from Triton
  CacheEntry lentry;
  lentry.triton_entry_ = entry;
  for (size_t item_index = 0; item_index < num_items; item_index++) {
    TRITONCACHE_CacheEntryItem* item = nullptr;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryGetItem(entry, item_index, &item));

    size_t num_buffers = 0;
    RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemBufferCount(item, &num_buffers));

    // Form cache representation of CacheEntryItem from Triton
    CacheEntryItem litem;
    litem.triton_item_ = item;
    for (size_t buffer_index = 0; buffer_index < num_buffers; buffer_index++) {
      // Get buffer and its buffer attributes from Triton
      void* base = nullptr;
      TRITONSERVER_BufferAttributes* attrs = nullptr;
      // TODO: Delete attrs on cache cleanup
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesNew(&attrs));
      RETURN_IF_ERROR(TRITONCACHE_CacheEntryItemGetBuffer(
          item, buffer_index, &base, attrs));

      // Query buffer attributes then clean up
      size_t byte_size = 0;
      TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t memory_type_id = 0;

      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesByteSize(attrs, &byte_size));
      RETURN_IF_ERROR(
          TRITONSERVER_BufferAttributesMemoryType(attrs, &memory_type));
      RETURN_IF_ERROR(
          TRITONSERVER_BufferAttributesMemoryTypeId(attrs, &memory_type_id));

      // DLIS-2673: Add better memory_type support
      if (memory_type != TRITONSERVER_MEMORY_CPU &&
          memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Only input buffers in CPU memory are allowed in cache currently");
      }

      if (!byte_size) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "buffer size was zero");
      }

      // TODO
      // Cache will replace this base pointer with a new cache-allocated base
      // pointer internally on Insert()
      litem.buffers_.emplace_back(std::make_pair(base, attrs));
    }
    lentry.items_.emplace_back(litem);
  }

  RETURN_IF_ERROR(lcache->Insert(key, lentry, allocator));
  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::cache::local
