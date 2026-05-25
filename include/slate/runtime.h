// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// runtime.h — L0: arena allocator, threadpool, logging.
//
// Everything below the tensor layer lives here. The single rule for this
// layer: zero allocation in the training loop's hot paths. Anything that
// looks like dynamic memory in a forward or backward op must come from an
// arena allocated up front.

#ifndef SLATE_RUNTIME_H
#define SLATE_RUNTIME_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Arena allocator
// =============================================================================
//
// An arena owns one contiguous buffer. Allocations from the arena are
// bump-pointer fast and never freed individually; the whole arena is reset
// in one call.
//
// Three arenas are typical in a training run:
//   - params  : alive for the whole run (weights + gradients)
//   - optim   : alive for the whole run (optimizer state)
//   - scratch : reset after every optimizer.step() (activations)
//
// Backing memory is OS-page-aligned for cache friendliness.

// Allocate a new arena with `capacity_bytes` of usable space. Returns NULL
// on out-of-memory.
slate_arena_t* slate_arena_create(size_t capacity_bytes);

// Free the arena and its backing buffer. Calling this while pointers into
// the arena are still in use is undefined behavior.
void slate_arena_destroy(slate_arena_t* arena);

// Allocate `bytes` aligned to `alignment` bytes from the arena. Alignment
// must be a power of two; pass 0 for default alignment (16 bytes, sufficient
// for SSE/NEON). Returns NULL if the arena is full.
void* slate_arena_alloc(slate_arena_t* arena, size_t bytes, size_t alignment);

// Returns the number of bytes currently in use.
size_t slate_arena_used(const slate_arena_t* arena);

// Returns the total capacity of the arena.
size_t slate_arena_capacity(const slate_arena_t* arena);

// Reset the arena's bump pointer to zero. All pointers obtained from this
// arena since the last reset are invalidated.
void slate_arena_reset(slate_arena_t* arena);

// =============================================================================
// Threadpool
// =============================================================================
//
// A fixed-size pool. M0 ships with a single-threaded stub (n_threads == 1);
// M3 swaps in a real work-stealing implementation. The interface here is
// stable across both.

typedef void (*slate_task_fn)(int task_id, int n_tasks, void* user_data);

// Create a pool with `n_threads` workers. Pass 0 to use the number of
// hardware threads. Returns NULL on failure.
slate_threadpool_t* slate_threadpool_create(int n_threads);

void slate_threadpool_destroy(slate_threadpool_t* pool);

// Returns the number of worker threads.
int slate_threadpool_num_threads(const slate_threadpool_t* pool);

// Adjusts the active worker count to `n`. Workers beyond `n` are kept
// alive but parked. Used by the ModeController to throttle for thermal
// pressure.
void slate_threadpool_set_active(slate_threadpool_t* pool, int n);

// Run `fn` `n_tasks` times in parallel across the pool. Blocks until all
// tasks complete. Safe to call from the main thread only.
void slate_threadpool_parallel_for(slate_threadpool_t* pool,
                                   int n_tasks,
                                   slate_task_fn fn,
                                   void* user_data);

// =============================================================================
// Logging
// =============================================================================

typedef enum slate_log_level {
    SLATE_LOG_TRACE = 0,
    SLATE_LOG_DEBUG = 1,
    SLATE_LOG_INFO  = 2,
    SLATE_LOG_WARN  = 3,
    SLATE_LOG_ERROR = 4,
    SLATE_LOG_OFF   = 5,
} slate_log_level_t;

// Set the global log level. Messages below this level are dropped.
void slate_log_set_level(slate_log_level_t level);

// Returns the current global log level.
slate_log_level_t slate_log_get_level(void);

// Internal logging entry point; usually invoked via the SLATE_LOG_* macros.
void slate_log_message(slate_log_level_t level, const char* file, int line,
                       const char* fmt, ...);

#define SLATE_LOG_INFO_F(fmt, ...) \
    slate_log_message(SLATE_LOG_INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define SLATE_LOG_WARN_F(fmt, ...) \
    slate_log_message(SLATE_LOG_WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define SLATE_LOG_ERROR_F(fmt, ...) \
    slate_log_message(SLATE_LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// Process-wide threadpool used by ops for parallel kernels.
slate_threadpool_t* slate_global_pool(void);

#ifdef __cplusplus
}
#endif

slate_threadpool_t* slate_global_pool(void);

#ifdef __cplusplus
}
#endif

#endif // SLATE_RUNTIME_H
