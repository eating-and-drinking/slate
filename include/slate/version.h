// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// version.h — compile-time and runtime version metadata.

#ifndef SLATE_VERSION_H
#define SLATE_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#define SLATE_VERSION_MAJOR 0
#define SLATE_VERSION_MINOR 1
#define SLATE_VERSION_PATCH 0

#define SLATE_VERSION_STRING "0.1.0"

// Numeric version suitable for runtime comparison: MMmmpp.
#define SLATE_VERSION_NUM ((SLATE_VERSION_MAJOR) * 10000 + \
                           (SLATE_VERSION_MINOR) * 100   + \
                           (SLATE_VERSION_PATCH))

// Returns a static string identifying the library build, e.g. "slate 0.1.0".
const char* slate_version_string(void);

// Returns the runtime numeric version. May not match the compile-time macros
// if a downstream binary is linked against a different libslate.
int slate_version_num(void);

#ifdef __cplusplus
}
#endif

#endif // SLATE_VERSION_H
