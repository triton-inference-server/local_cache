<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Report a Security Vulnerability

To report a potential security vulnerability in this project or any NVIDIA
product, please use one of the following channels. **Do not open a public
GitHub issue for a suspected vulnerability.**

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   [https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/)
2. **Email NVIDIA PSIRT:** [psirt@nvidia.com](mailto:psirt@nvidia.com).
   Please encrypt sensitive reports with NVIDIA's
   [PGP key](https://www.nvidia.com/en-us/security/pgp-key).
3. **GitHub Private Vulnerability Reporting:** use the "Report a vulnerability"
   button on the repository's Security tab, where enabled.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (memory corruption, denial of service, information
   disclosure, etc.)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, triages and validates them, coordinates a
fix and disclosure timeline with the reporter, and publishes security
bulletins at [https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).

# Security Architecture and Context

**Project:** Triton Local Cache, the in-memory response cache implementation
for [Triton Inference Server](https://github.com/triton-inference-server/server).

**Software classification:** Library. It is a C++17 shared library
(`libtritoncache_local.so`) that Triton Core loads as a cache plugin through
the `TRITONCACHE_*` C API (`src/cache_api.cc`). It is not a standalone process
and opens no network listeners, files, or sockets of its own.

**Primary security responsibility:** Store and return cached inference
response buffers faithfully and within a fixed memory budget, without memory
corruption, cross-entry data mixing, or unbounded resource use.

**Key interfaces and boundaries:**

- **Plugin API (trust boundary with Triton Core):** `CacheInitialize`,
  `CacheFinalize`, `CacheLookup`, `CacheInsert`. Triton Core supplies the
  cache key, entry buffers, buffer attributes, and the copy allocator. The
  library validates for null arguments, zero-sized buffers, and CPU or pinned
  CPU memory type only.
- **Configuration input:** a JSON string passed by the server operator
  (`--cache-config local,size=<bytes>`), parsed with RapidJSON in
  `LocalCache::Create`. Only the `size` field is consumed.
- **Memory:** one `malloc`-ed region of the configured size, managed with
  `boost::interprocess::managed_external_buffer` and guarded by internal
  mutexes. Eviction is least-recently-used.
- **Observability:** optional Prometheus metrics (`nv_cache_*`) through the
  Triton metrics API, and Triton logging.

**Repository Exposure Classification:** Public. Basis: the repository is
publicly visible on GitHub.

**Service Exposure Classification:** Internal-Sensitive (medium confidence).
Basis: the library runs inside the Triton server process and handles model
inference responses, which can contain customer data, but it has no direct
network exposure and inherits the deployment's exposure from the host server.

# Threat Model

1. **Resource exhaustion through cache sizing:** the `size` setting is
   allocated in one `malloc` call at initialization. An excessively large or
   malformed value can cause start-up failure or memory pressure on the host.
   Treat the cache configuration as trusted operator input.
2. **Cross-request data exposure through shared cache entries:** all
   callers of one cache instance share a single key space, and entries are
   returned to any caller presenting the same key. If Triton Core, or a
   deployment, derives keys without covering all inputs that affect the
   response (model, version, input data, relevant parameters), one request can
   receive another's cached output.
3. **Memory-safety defects in buffer lifetime handling:** the library stores
   raw pointers into the managed buffer, LRU list iterators, and per-entry
   attribute objects, and relies on Triton Core copying data out before an
   entry can be evicted. Bugs in eviction, entry lifetime, or locking
   (`cache_mu_` and `buffer_mu_` ordering) could lead to use-after-free,
   double free, or data races. Concurrency changes should be reviewed with
   this in mind.
4. **Denial of service through cache churn:** insertion of many or large
   entries evicts useful entries and can fail allocation when a single
   response exceeds the cache size or memory is fragmented. Callers must treat
   insert failures as non-fatal and fall back to uncached execution.
5. **Information disclosure through logs and metrics:** the cache
   configuration and cache keys appear in log messages, and aggregate
   utilization, hit, and eviction counters are exported as metrics. Keys
   derived from request data should not be assumed confidential in logs.
6. **Supply-chain risk in build dependencies:** the library builds against
   Boost, RapidJSON, and Triton common and core components. Compromised or
   vulnerable dependency versions would affect the resulting binary.

# Critical Security Assumptions

- **No authentication or authorization:** the library performs none. Access
  control is entirely the responsibility of Triton Server and its deployment.
- **Trusted caller:** Triton Core is assumed to pass valid buffers, correct
  sizes, and keys that fully identify the response being cached.
- **Trusted configuration:** the cache configuration string is assumed to come
  from the server operator, not from untrusted users.
- **No encryption at rest or in memory:** cached responses are stored as
  plaintext in process memory and are not zeroed on eviction, so other code in
  the same process (or a memory dump of it) can observe them.
- **Single-tenant cache:** there is no per-tenant isolation or quota. Do not
  share one cache instance between mutually distrusting tenants.
- **CPU memory only:** only CPU and pinned CPU buffers are supported. GPU
  memory buffers are rejected.
- **Process isolation:** the host process boundary and operating system
  protections are assumed to prevent untrusted code from reading or writing
  the cache region.
