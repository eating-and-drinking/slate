// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 02_mnist — train a 3-layer MLP on MNIST.
//
// Usage:
//   slate_mnist <data_dir>
// where <data_dir> contains the four IDX files:
//   train-images-idx3-ubyte
//   train-labels-idx1-ubyte
//   t10k-images-idx3-ubyte
//   t10k-labels-idx1-ubyte
//
// Get them from http://yann.lecun.com/exdb/mnist/ (the original distribution
// is intermittently down; HuggingFace and other mirrors carry it).
//
// Architecture: 784 -> 128 -> 64 -> 10, ReLU between layers, cross-entropy.
// Acceptance criterion: test accuracy >= 97%.

#include "slate/slate.h"
#include "slate/lr_scheduler.h"
#include "slate/data_simple.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_DIM    784
#define H1_DIM       128
#define H2_DIM        64
#define N_CLASSES     10
#define BATCH_SIZE   128
#define N_EPOCHS      10
#define LR_MAX     5e-4f
#define LR_MIN     1e-5f

static char* join_path(const char* dir, const char* name) {
    size_t n = strlen(dir) + 1 + strlen(name) + 1;
    char* p = (char*)malloc(n);
    snprintf(p, n, "%s/%s", dir, name);
    return p;
}

static void images_to_f32(const slate_idx_data_t* idx, float* out) {
    int n = idx->n_items;
    int sz = idx->rows * idx->cols;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < sz; ++j) {
            out[i * sz + j] = (float)idx->data[i * sz + j] / 255.0f;
        }
    }
}

static void labels_to_i32(const slate_idx_data_t* idx, int32_t* out) {
    for (int i = 0; i < idx->n_items; ++i) out[i] = (int32_t)idx->data[i];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mnist-dir>\n", argv[0]);
        fprintf(stderr, "  <mnist-dir> must contain the four IDX files.\n");
        return 2;
    }
    printf("slate %s — mnist example\n", SLATE_VERSION_STRING);

    // -------------------------------------------------------------------------
    // Load data.
    // -------------------------------------------------------------------------
    slate_idx_data_t train_x_idx, train_y_idx, test_x_idx, test_y_idx;
    char* p_tx = join_path(argv[1], "train-images-idx3-ubyte");
    char* p_ty = join_path(argv[1], "train-labels-idx1-ubyte");
    char* p_ex = join_path(argv[1], "t10k-images-idx3-ubyte");
    char* p_ey = join_path(argv[1], "t10k-labels-idx1-ubyte");
    if (slate_idx_load_images(p_tx, &train_x_idx) != SLATE_OK ||
        slate_idx_load_labels(p_ty, &train_y_idx) != SLATE_OK ||
        slate_idx_load_images(p_ex, &test_x_idx)  != SLATE_OK ||
        slate_idx_load_labels(p_ey, &test_y_idx)  != SLATE_OK) {
        fprintf(stderr, "MNIST load failed: %s\n", slate_last_error());
        return 1;
    }
    free(p_tx); free(p_ty); free(p_ex); free(p_ey);
    printf("[mnist] train: %d images, test: %d images\n",
           train_x_idx.n_items, test_x_idx.n_items);

    float*   X_train = (float*)malloc((size_t)train_x_idx.n_items * INPUT_DIM * sizeof(float));
    int32_t* y_train = (int32_t*)malloc((size_t)train_x_idx.n_items * sizeof(int32_t));
    float*   X_test  = (float*)malloc((size_t)test_x_idx.n_items  * INPUT_DIM * sizeof(float));
    int32_t* y_test  = (int32_t*)malloc((size_t)test_x_idx.n_items  * sizeof(int32_t));
    images_to_f32(&train_x_idx, X_train);
    labels_to_i32(&train_y_idx, y_train);
    images_to_f32(&test_x_idx,  X_test);
    labels_to_i32(&test_y_idx,  y_test);
    int n_train = train_x_idx.n_items;
    int n_test  = test_x_idx.n_items;
    slate_idx_free(&train_x_idx); slate_idx_free(&train_y_idx);
    slate_idx_free(&test_x_idx);  slate_idx_free(&test_y_idx);

    // -------------------------------------------------------------------------
    // Build model and optimizer.
    // -------------------------------------------------------------------------
    slate_arena_t* params  = slate_arena_create( 32 * 1024 * 1024);
    slate_arena_t* opt_st  = slate_arena_create( 64 * 1024 * 1024);
    slate_arena_t* nodes   = slate_arena_create( 16 * 1024 * 1024);
    slate_arena_t* scratch = slate_arena_create(128 * 1024 * 1024);

    slate_module_t* fc1 = slate_module_linear_new(params, INPUT_DIM, H1_DIM,    true, 1);
    slate_module_t* fc2 = slate_module_linear_new(params, H1_DIM,    H2_DIM,    true, 2);
    slate_module_t* fc3 = slate_module_linear_new(params, H2_DIM,    N_CLASSES, true, 3);

    slate_param_set_t ps;
    slate_param_set_init(&ps);
    slate_module_register_params(fc1, &ps);
    slate_module_register_params(fc2, &ps);
    slate_module_register_params(fc3, &ps);
    printf("[mnist] %d parameter tensors\n", ps.n_params);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(opt_st, &ps,
                                                       LR_MAX, 0.9f, 0.999f,
                                                       1e-8f, 0.01f);

    int n_batches = n_train / BATCH_SIZE;
    int total_steps = n_batches * N_EPOCHS;
    int warmup_steps = total_steps / 50;
    slate_lr_scheduler_t* sched = slate_lr_cosine_warmup_new(LR_MAX, LR_MIN,
                                                              warmup_steps,
                                                              total_steps);

    slate_simple_dataloader_t* dl = slate_simple_dataloader_new(n_train,
                                                                BATCH_SIZE,
                                                                true, 0xA5A5);
    int batch_idx[BATCH_SIZE];
    int64_t xshape[2] = {BATCH_SIZE, INPUT_DIM};
    int64_t yshape[1] = {BATCH_SIZE};

    // -------------------------------------------------------------------------
    // Train.
    // -------------------------------------------------------------------------
    int global_step = 0;
    clock_t t0 = clock();
    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        slate_simple_dataloader_reset(dl);
        double epoch_loss = 0;
        int seen = 0;

        while (slate_simple_dataloader_next(dl, batch_idx)) {
            float lr = slate_lr_scheduler_get(sched, global_step);
            slate_optimizer_set_lr(opt, lr);

            slate_graph_ctx_t ctx;
            slate_graph_ctx_init(&ctx, nodes, scratch);
            ctx.training = true;

            slate_tensor_t* xb = slate_tensor_new(scratch, SLATE_DTYPE_F32, 2, xshape, false);
            slate_tensor_t* yb = slate_tensor_new(scratch, SLATE_DTYPE_I32, 1, yshape, false);
            float* xb_p = (float*)xb->data;
            int32_t* yb_p = (int32_t*)yb->data;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                memcpy(xb_p + i * INPUT_DIM, X_train + batch_idx[i] * INPUT_DIM,
                       INPUT_DIM * sizeof(float));
                yb_p[i] = y_train[batch_idx[i]];
            }

            slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, xb);
            slate_tensor_t* a1 = slate_op_relu(&ctx, h1);
            slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
            slate_tensor_t* a2 = slate_op_relu(&ctx, h2);
            slate_tensor_t* logits = slate_module_forward(fc3, &ctx, a2);
            slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, logits, yb);
            epoch_loss += ((const float*)loss->data)[0];
            seen++;

            slate_optimizer_zero_grad(opt);
            slate_graph_backward(&ctx, loss);
            slate_clip_grad_norm(&ps, 1.0f);
            slate_optimizer_step(opt);
            slate_graph_ctx_reset(&ctx);
            global_step++;
        }

        // Test set evaluation.
        int correct = 0;
        int n_test_batches = n_test / BATCH_SIZE;
        for (int bi = 0; bi < n_test_batches; ++bi) {
            slate_graph_ctx_t ctx;
            slate_graph_ctx_init(&ctx, nodes, scratch);
            ctx.training = false;

            slate_tensor_t* xb = slate_tensor_new(scratch, SLATE_DTYPE_F32, 2, xshape, false);
            memcpy(xb->data,
                   X_test + bi * BATCH_SIZE * INPUT_DIM,
                   BATCH_SIZE * INPUT_DIM * sizeof(float));

            slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, xb);
            slate_tensor_t* a1 = slate_op_relu(&ctx, h1);
            slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
            slate_tensor_t* a2 = slate_op_relu(&ctx, h2);
            slate_tensor_t* logits = slate_module_forward(fc3, &ctx, a2);

            const float* p = (const float*)logits->data;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                int best = 0;
                float bv = p[i * N_CLASSES];
                for (int c = 1; c < N_CLASSES; ++c) {
                    if (p[i * N_CLASSES + c] > bv) { bv = p[i * N_CLASSES + c]; best = c; }
                }
                if (best == y_test[bi * BATCH_SIZE + i]) correct++;
            }
            slate_graph_ctx_reset(&ctx);
        }
        float acc = (float)correct / (float)(n_test_batches * BATCH_SIZE);
        printf("[mnist] epoch %2d  loss=%.4f  test_acc=%.4f\n",
               epoch, epoch_loss / seen, acc);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("[mnist] done in %.1fs\n", secs);

    free(X_train); free(y_train); free(X_test); free(y_test);
    slate_simple_dataloader_destroy(dl);
    slate_lr_scheduler_destroy(sched);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(fc1);
    slate_module_destroy(fc2);
    slate_module_destroy(fc3);
    slate_arena_destroy(params);
    slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes);
    slate_arena_destroy(scratch);
    return 0;
}
