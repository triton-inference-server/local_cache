<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Local Cache

This repo contains an example
[TRITONCACHE API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritoncache.h)
implementation for caching data locally in-memory.

Ask questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues).

## Build the Cache

Use a recent cmake to build. First install the required dependencies.

```
$ apt-get install libboost-dev rapidjson-dev
```

To build the cache:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the following CMake arguments can be used to override.

* triton-inference-server/core: `-D TRITON_CORE_REPO_TAG=[tag]`
* triton-inference-server/common: `-D TRITON_COMMON_REPO_TAG=[tag]`

## Configuring the Cache

Like other `TRITONCACHE` implementations, this cache is configured through the
`tritonserver --cache-config` CLI arg or through the
`TRITONSERVER_SetCacheConfig` API.

Currently, the following config fields are supported:
- `size`: The fixed size (in bytes) of CPU memory allocated to the cache
upfront. If this value is too large (ex: greater than available memory) or
too small (ex: smaller than required overhead such as ~1-2 KB), initialization
may fail.
    - example: `tritonserver --cache-config local,size=1048576`

## Metrics

When `TRITON_ENABLE_METRICS` is enabled in this cache (enabled by default),
it will check to see if the running Triton server has metrics enabled as well.
If so, the cache will publish additional cache-specific metrics to Triton's
metrics endpoint through the
[Custom Metrics API](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md#custom-metrics).

### Cache Metrics

The following metrics are reported by this cache implementation:

|Category      |Metric                     |Metric Name                   |Description                                                 |Granularity |Frequency    |
|--------------|---------------------------|------------------------------|------------------------------------------------------------|------------|-------------|
|Utilization   |Total Cache Utilization    |`nv_cache_util`               |Total cache utilization rate (0.0 - 1.0)                    |Server-wide |Per interval |
|Count         |Total Cache Entry Count    |`nv_cache_num_entries`        |Total number of entries stored in cache                     |Server-wide |Per interval |
|              |Total Cache Lookup Count   |`nv_cache_num_lookups`        |Total number of cache lookups done by Triton                |Server-wide |Per interval |
|              |Total Cache Hit Count      |`nv_cache_num_hits`           |Total number of cache hits                                  |Server-wide |Per interval |
|              |Total Cache Miss Count     |`nv_cache_num_misses`         |Total number of cache misses                                |Server-wide |Per interval |
|              |Total Cache Eviction Count |`nv_cache_num_evictions`      |Total number of cache evictions                             |Server-wide |Per interval |
|Latency       |Total Cache Lookup Time    |`nv_cache_lookup_duration`    |Cumulative time spent doing cache lookups (microseconds)    |Server-wide |Per interval |
|              |Total Cache Insertion Time |`nv_cache_insertion_duration` |Cumulative time spent doint cache insertions (microseconds) |Server-wide |Per interval |


