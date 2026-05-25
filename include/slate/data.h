// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// data.h — data plane: conversation logging, preference mining, batching.
//
// IMPLEMENTATION STATUS: M5+.

#ifndef SLATE_DATA_H
#define SLATE_DATA_H

#include "slate/types.h"
#include "slate/objective.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ConversationLogger: append-only JSONL of user/assistant interactions.
// =============================================================================

typedef struct slate_conversation_logger slate_conversation_logger_t;

slate_conversation_logger_t* slate_conversation_logger_new(const char* jsonl_path);
void slate_conversation_logger_destroy(slate_conversation_logger_t* lg);

slate_status_t slate_conversation_log_user(slate_conversation_logger_t* lg,
                                            const char* text);
slate_status_t slate_conversation_log_assistant(slate_conversation_logger_t* lg,
                                                 const char* msg_id,
                                                 const char* text);
slate_status_t slate_conversation_log_feedback(slate_conversation_logger_t* lg,
                                                const char* msg_id,
                                                int feedback_signal);  // +1 / -1
slate_status_t slate_conversation_log_edit(slate_conversation_logger_t* lg,
                                            const char* msg_id,
                                            const char* edited_text);

// =============================================================================
// PreferencePairBuilder: mines KTO/DPO data from the day's log.
// =============================================================================

typedef struct slate_preference_builder slate_preference_builder_t;

slate_preference_builder_t* slate_preference_builder_new(const char* jsonl_path);
void slate_preference_builder_destroy(slate_preference_builder_t* pb);

// Extract KTO-format (response, binary_label) examples.
slate_status_t slate_preference_builder_extract_kto(slate_preference_builder_t* pb,
                                                     const char* out_path);

// Extract DPO-format (chosen, rejected) pairs.
slate_status_t slate_preference_builder_extract_dpo(slate_preference_builder_t* pb,
                                                     const char* out_path);

// =============================================================================
// DataLoader: yields batches to the trainer.
// =============================================================================

typedef struct slate_dataloader slate_dataloader_t;

slate_dataloader_t* slate_dataloader_new(const char* dataset_path,
                                          int batch_size,
                                          int seq_len,
                                          bool shuffle,
                                          uint64_t seed);
void slate_dataloader_destroy(slate_dataloader_t* dl);

slate_status_t slate_dataloader_next(slate_dataloader_t* dl, slate_batch_t* batch);

// MixedDataLoader: multi-source weighted sampling.
typedef struct slate_dataloader_source {
    slate_dataloader_t* loader;
    float weight;
} slate_dataloader_source_t;

slate_dataloader_t* slate_dataloader_mixed_new(const slate_dataloader_source_t* sources,
                                                int n_sources,
                                                uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif // SLATE_DATA_H
