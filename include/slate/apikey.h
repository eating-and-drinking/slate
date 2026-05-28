// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// apikey.h — multi-key auth + per-key token-bucket rate limiting.
//
// Each registered key carries:
//   * the secret string (matched against Bearer header);
//   * a token bucket: capacity = `burst`, refill = `rps` tokens/sec;
//   * an optional human label (used in metrics and logs).
//
// On a request we lookup by secret, then take one token from the
// bucket — if the bucket is empty the request is rejected with 429.
// Thread-safe: one mutex per key (so different keys don't contend).
//
// Keys can be:
//   * added programmatically with slate_apikey_set_add(); or
//   * loaded from a JSON file with slate_apikey_set_load(): an array
//     of objects `[{"key":"...","label":"...","rps":N,"burst":M}, ...]`.
//
// The set is owned by the caller and must outlive the server.

#ifndef SLATE_APIKEY_H
#define SLATE_APIKEY_H

#include "slate/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_apikey_set slate_apikey_set_t;

slate_apikey_set_t* slate_apikey_set_new(void);
void                slate_apikey_set_free(slate_apikey_set_t* set);

// Returns 0 on success, < 0 on capacity/dup/invalid input.
//   key   : bearer secret (copied)
//   label : human-readable name for logs/metrics (copied); may be NULL
//   rps   : refill rate (tokens per second); 0 disables rate limit
//   burst : bucket capacity (max simultaneous tokens); 0 disables rate limit
int slate_apikey_set_add(slate_apikey_set_t* set,
                          const char* key,
                          const char* label,
                          double rps,
                          int    burst);

// Parse a JSON file into the key set.  Returns the number of keys
// loaded, or < 0 on parse error.  Lines starting with '#' are skipped.
// Schema: array of objects with fields {key, label?, rps?, burst?}.
int slate_apikey_set_load(slate_apikey_set_t* set, const char* path);

// Number of registered keys.
int slate_apikey_set_size(const slate_apikey_set_t* set);

// Check + consume a token.
//
//   set      : the registered key set
//   bearer   : the value after "Bearer " in the Authorization header
//   out_label: on success, set to the key's label (or its secret
//              prefix if label was NULL) — pointer is owned by set,
//              valid for the lifetime of the set
//
// Returns:
//    0  : authorised and within rate limit (one token consumed)
//   -1  : bearer not recognised (HTTP 401)
//   -2  : recognised but bucket empty (HTTP 429)
//
// Thread-safe.
int slate_apikey_check(slate_apikey_set_t* set,
                        const char* bearer,
                        const char** out_label);

#ifdef __cplusplus
}
#endif

#endif // SLATE_APIKEY_H
