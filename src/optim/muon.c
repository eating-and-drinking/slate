// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// muon.c — Muon optimizer (Jordan et al., 2024).
//
// Muon = MomentUm Orthogonalized by Newton-Schulz.  For 2D parameter
// matrices it maintains a momentum buffer like SGD, then orthogonalises
// the (Nesterov-shifted) update via a fixed-coefficient quintic
// Newton-Schulz iteration before applying it.  This makes every
// gradient direction "matter equally" along all left/right singular
// vectors, which empirically gives faster convergence per FLOP on
// LLM training than Adam/AdamW.
//
// For 1D / 3D+ parameters (biases, embeddings table, conv tensors,
// LayerNorm weights, etc.) Muon does NOT NS-orthogonalise — those
// shapes are not square-ish enough to make the iteration meaningful —
// and falls back to a plain SGD-with-momentum step.  This matches the
// reference Python implementation, where users typically pair Muon
// (on the matrix params) with AdamW (on everything else).  Slate's
// flavour keeps both behaviours under one optimizer for ergonomics.
//
// Algorithm per parameter, every step:
//   m_t      = β·m_{t-1} + g_t                       (momentum)
//   if 2D:
//       u_t  = g_t + β·m_t                            (Nesterov shift)
//       u_t  = NS5(u_t / ||u_t||_F) · max(1, √(M/N))  (orthogonalise + dim scale)
//   else:
//       u_t  = m_t
//   w   -= lr·wd·w                                    (decoupled weight decay)
//   w   -= lr·u_t
//
// Newton-Schulz quintic (Keller Jordan's coefficients):
//   X ← X / ||X||_F
//   for s in 1..ns_steps:
//       A = X · Xᵀ
//       B = b·A + c·A·A
//       X = a·X + B·X
// with (a, b, c) = (3.4445, -4.7750, 2.0315).  Five iterations suffice
// to push every singular value of X into ~[0.8, 1.2] regardless of
// starting condition, which is "orthogonal enough" for SGD.
//
// All matmuls in NS5 dispatch to the same packed-panel AVX2 GEMM that
// powers slate_op_matmul/bmm — so Muon's per-step cost on a 4096×4096
// weight on this machine is ~5·(O(n³)) ≈ 1.7 GFLOP at ~60 GFLOP/s ≈
// 28 ms, which is dominated by the forward+backward pass for any
// non-trivial network and so does not change overall step time
// materially.

#include "slate/optim.h"
#include "slate/tensor.h"
#include "slate/module.h"
#include "../ops/gemm_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Per-parameter and global state
// ---------------------------------------------------------------------------

typedef struct muon_param {
    float*  m;        // momentum buffer, same numel as the parameter
    int     is_2d;
    int64_t rows;     // shape[0]    (only valid if is_2d)
    int64_t cols;     // shape[1]    (only valid if is_2d)
} muon_param_t;

typedef struct muon_state {
    slate_param_set_t* params;
    float   lr;
    float   beta;          // momentum coefficient
    float   weight_decay;
    int     ns_steps;
    int     step_count;
    muon_param_t* per;

    // Shared NS scratch.  Sized once at construction to the largest 2D
    // parameter seen, since every NS iteration writes/reads transiently
    // and we don't need per-parameter copies.
    float*  scratch_X;     // [max_numel_2d]   working matrix (and post-transpose)
    float*  scratch_Xnew;  // [max_numel_2d]   B·X scratch + transpose ping-pong
    float*  scratch_A;     // [max_small_sq]   X·Xᵀ  (and then b·A + c·A²)
    float*  scratch_A2;    // [max_small_sq]   A·A
} muon_state_t;

// ---------------------------------------------------------------------------
// Small AVX2 helpers
// ---------------------------------------------------------------------------

// dst[i] = beta * dst[i] + src[i]      (momentum update; in-place)
static inline void vec_momentum_update(float* dst, const float* src,
                                       float beta, int64_t n) {
#if defined(__AVX2__)
    __m256 bv = _mm256_set1_ps(beta);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(bv, d, s));
    }
    for (; i < n; ++i) dst[i] = beta * dst[i] + src[i];
#else
    for (int64_t i = 0; i < n; ++i) dst[i] = beta * dst[i] + src[i];
#endif
}

// dst[i] = g[i] + beta * m[i]          (Nesterov-shifted update direction)
static inline void vec_nesterov(float* dst, const float* g, const float* m,
                                float beta, int64_t n) {
#if defined(__AVX2__)
    __m256 bv = _mm256_set1_ps(beta);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 gv = _mm256_loadu_ps(g + i);
        __m256 mv = _mm256_loadu_ps(m + i);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(bv, mv, gv));
    }
    for (; i < n; ++i) dst[i] = g[i] + beta * m[i];
#else
    for (int64_t i = 0; i < n; ++i) dst[i] = g[i] + beta * m[i];
#endif
}

// dst[i] *= s
static inline void vec_scale_inplace(float* dst, float s, int64_t n) {
#if defined(__AVX2__)
    __m256 sv = _mm256_set1_ps(s);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(d, sv));
    }
    for (; i < n; ++i) dst[i] *= s;
#else
    for (int64_t i = 0; i < n; ++i) dst[i] *= s;
#endif
}

// w[i] -= lr * scale * u[i]
static inline void vec_apply_update(float* w, const float* u,
                                    float lr, float scale, int64_t n) {
    float k = lr * scale;
#if defined(__AVX2__)
    __m256 kv = _mm256_set1_ps(k);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 wv = _mm256_loadu_ps(w + i);
        __m256 uv = _mm256_loadu_ps(u + i);
        _mm256_storeu_ps(w + i, _mm256_fnmadd_ps(kv, uv, wv));
    }
    for (; i < n; ++i) w[i] -= k * u[i];
#else
    for (int64_t i = 0; i < n; ++i) w[i] -= k * u[i];
#endif
}

// Frobenius norm (double accumulator — orthogonality is sensitive to
// the scale, so don't lose precision on big matrices).
static double frobenius_norm(const float* X, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double v = (double)X[i];
        s += v * v;
    }
    return sqrt(s);
}

// dst = a * X + B · X       (B is rows×rows, X is rows×cols; both row-major)
static void linear_combine(float* dst, float a,
                            const float* X, const float* B,
                            int rows, int cols) {
    int64_t n = (int64_t)rows * cols;
    // dst = a * X
#if defined(__AVX2__)
    __m256 av = _mm256_set1_ps(a);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(X + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(xv, av));
    }
    for (; i < n; ++i) dst[i] = a * X[i];
#else
    for (int64_t i = 0; i < n; ++i) dst[i] = a * X[i];
#endif
    // dst += B · X
    slate_gemm_packed_accumulate(dst, /*ldc=*/cols,
                                  B,   /*lda=*/rows,
                                  X,   /*ldb=*/cols,
                                  rows, rows, cols);
}

// ---------------------------------------------------------------------------
// Newton-Schulz 5-term quintic.
// ---------------------------------------------------------------------------
//
// Operates on X (rows × cols) with the precondition that rows ≤ cols
// (caller transposes if necessary so the small dimension is on the
// left).  X is overwritten in-place with the orthogonalised result.
// Scratch buffers must be pre-allocated on muon_state_t.

static void newton_schulz5(muon_state_t* st, float* X,
                            int rows, int cols, int steps) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    float* A   = st->scratch_A;
    float* A2  = st->scratch_A2;
    float* Xn  = st->scratch_Xnew;

    int64_t numel = (int64_t)rows * cols;
    int64_t numA  = (int64_t)rows * rows;

    for (int s = 0; s < steps; ++s) {
        // A = X · Xᵀ  (rows × rows).  Materialise Xᵀ into Xn first since
        // the packed GEMM only takes row-major non-transposed inputs.
        slate_transpose_f32(X, Xn, rows, cols);  // Xn = Xᵀ, shape cols×rows
        memset(A, 0, (size_t)numA * sizeof(float));
        slate_gemm_packed_accumulate(A,  /*ldc=*/rows,
                                      X,  /*lda=*/cols,
                                      Xn, /*ldb=*/rows,
                                      rows, cols, rows);

        // A2 = A · A
        memset(A2, 0, (size_t)numA * sizeof(float));
        slate_gemm_packed_accumulate(A2, /*ldc=*/rows,
                                      A,  /*lda=*/rows,
                                      A,  /*ldb=*/rows,
                                      rows, rows, rows);

        // A ← b·A + c·A²   (so A now plays the role of "B" in the iteration)
#if defined(__AVX2__)
        {
            __m256 bv = _mm256_set1_ps(b);
            __m256 cv = _mm256_set1_ps(c);
            int64_t i = 0;
            for (; i + 8 <= numA; i += 8) {
                __m256 av = _mm256_loadu_ps(A + i);
                __m256 a2v= _mm256_loadu_ps(A2 + i);
                __m256 r  = _mm256_fmadd_ps(bv, av, _mm256_mul_ps(cv, a2v));
                _mm256_storeu_ps(A + i, r);
            }
            for (; i < numA; ++i) A[i] = b * A[i] + c * A2[i];
        }
#else
        for (int64_t i = 0; i < numA; ++i) A[i] = b * A[i] + c * A2[i];
#endif

        // Xn = a·X + A · X
        linear_combine(Xn, a, X, A, rows, cols);

        // X ← Xn
        memcpy(X, Xn, (size_t)numel * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Optimizer vtable methods
// ---------------------------------------------------------------------------

static void muon_step(slate_optimizer_t* self) {
    muon_state_t* st = (muon_state_t*)self->user_data;
    st->step_count++;

    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_t* t = st->params->params[p];
        if (!t || !t->grad) continue;
        float*       w = (float*)t->data;
        const float* g = (const float*)t->grad;
        muon_param_t* mp = &st->per[p];
        int64_t n = slate_tensor_numel(t);

        // Decoupled weight decay (AdamW-style).
        if (st->weight_decay > 0.0f) {
            float wd = st->lr * st->weight_decay;
#if defined(__AVX2__)
            __m256 wdv = _mm256_set1_ps(wd);
            int64_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 wv = _mm256_loadu_ps(w + i);
                _mm256_storeu_ps(w + i, _mm256_fnmadd_ps(wdv, wv, wv));
            }
            for (; i < n; ++i) w[i] -= wd * w[i];
#else
            for (int64_t i = 0; i < n; ++i) w[i] -= wd * w[i];
#endif
        }

        // Momentum: m = β·m + g
        vec_momentum_update(mp->m, g, st->beta, n);

        if (!mp->is_2d) {
            // Fallback: SGD-with-momentum on whatever shape this is.
            vec_apply_update(w, mp->m, st->lr, 1.0f, n);
            continue;
        }

        // 2D path.  Build Nesterov-shifted update direction in scratch_X.
        float* X = st->scratch_X;
        vec_nesterov(X, g, mp->m, st->beta, n);

        // NS expects the small dimension on the left (rows ≤ cols).
        // If the parameter is "tall" (M > N), operate on the transpose
        // and undo at the end.
        int64_t M = mp->rows, N_ = mp->cols;
        int     rows = (int)M, cols = (int)N_;
        int     transposed = 0;
        if (M > N_) {
            slate_transpose_f32(X, st->scratch_Xnew, (int)M, (int)N_);
            memcpy(X, st->scratch_Xnew, (size_t)n * sizeof(float));
            rows = (int)N_;
            cols = (int)M;
            transposed = 1;
        }

        // Normalise: NS5 needs ||X||_F ≈ 1 to stay inside its convergence basin.
        double fnorm = frobenius_norm(X, n);
        if (fnorm < 1e-7) {
            // Update direction is essentially zero — skip the orthogonalisation
            // and the step.  (The momentum buffer was already updated above.)
            continue;
        }
        vec_scale_inplace(X, (float)(1.0 / (fnorm + 1e-7)), n);

        // Run the quintic iteration.
        newton_schulz5(st, X, rows, cols, st->ns_steps);

        // Apply dim scaling.  Per Keller Jordan: max(1, √(rows / cols))
        // in terms of the ORIGINAL parameter shape (before any transpose).
        float dim_scale = sqrtf((float)M / (float)N_);
        if (dim_scale < 1.0f) dim_scale = 1.0f;

        // Apply the update.  If we transposed, undo so X is back to M×N_
        // before subtracting from w.
        if (transposed) {
            slate_transpose_f32(X, st->scratch_Xnew, rows, cols);
            vec_apply_update(w, st->scratch_Xnew, st->lr, dim_scale, n);
        } else {
            vec_apply_update(w, X, st->lr, dim_scale, n);
        }
    }
}

static void muon_zero_grad(slate_optimizer_t* self) {
    muon_state_t* st = (muon_state_t*)self->user_data;
    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_zero_grad(st->params->params[p]);
    }
}

static void muon_set_lr(slate_optimizer_t* self, float lr) {
    muon_state_t* st = (muon_state_t*)self->user_data;
    st->lr = lr;
}

static void muon_destroy(slate_optimizer_t* self) {
    if (!self) return;
    muon_state_t* st = (muon_state_t*)self->user_data;
    if (st) {
        free(st->per);
        free(st);
    }
    free(self);
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

slate_optimizer_t* slate_optimizer_muon_new(slate_arena_t* state_arena,
                                             slate_param_set_t* params,
                                             float learning_rate,
                                             float momentum,
                                             float weight_decay,
                                             int   ns_steps) {
    if (!params) return NULL;
    if (ns_steps <= 0) ns_steps = 5;

    slate_optimizer_t* opt = (slate_optimizer_t*)calloc(1, sizeof(*opt));
    muon_state_t*      st  = (muon_state_t*)calloc(1, sizeof(*st));
    if (!opt || !st) { free(opt); free(st); return NULL; }

    st->params       = params;
    st->lr           = learning_rate;
    st->beta         = momentum;
    st->weight_decay = weight_decay;
    st->ns_steps     = ns_steps;
    st->step_count   = 0;
    st->per = (muon_param_t*)calloc((size_t)params->n_params, sizeof(muon_param_t));

    int64_t max_numel_2d = 0;
    int64_t max_small_sq = 0;
    for (int p = 0; p < params->n_params; ++p) {
        slate_tensor_t* t = params->params[p];
        muon_param_t* mp = &st->per[p];
        int64_t n = slate_tensor_numel(t);
        mp->m = (float*)slate_arena_alloc(state_arena,
                                           (size_t)n * sizeof(float), 16);
        if (mp->m) memset(mp->m, 0, (size_t)n * sizeof(float));
        if (t->n_dims == 2) {
            mp->is_2d = 1;
            mp->rows  = t->shape[0];
            mp->cols  = t->shape[1];
            if (n > max_numel_2d) max_numel_2d = n;
            int64_t small = mp->rows < mp->cols ? mp->rows : mp->cols;
            int64_t small_sq = small * small;
            if (small_sq > max_small_sq) max_small_sq = small_sq;
        } else {
            mp->is_2d = 0;
            mp->rows  = 0;
            mp->cols  = 0;
        }
    }

    if (max_numel_2d > 0) {
        st->scratch_X    = (float*)slate_arena_alloc(state_arena,
                                                      (size_t)max_numel_2d * sizeof(float), 16);
        st->scratch_Xnew = (float*)slate_arena_alloc(state_arena,
                                                      (size_t)max_numel_2d * sizeof(float), 16);
        st->scratch_A    = (float*)slate_arena_alloc(state_arena,
                                                      (size_t)max_small_sq * sizeof(float), 16);
        st->scratch_A2   = (float*)slate_arena_alloc(state_arena,
                                                      (size_t)max_small_sq * sizeof(float), 16);
    }

    opt->step      = muon_step;
    opt->zero_grad = muon_zero_grad;
    opt->set_lr    = muon_set_lr;
    opt->destroy   = muon_destroy;
    opt->user_data = st;
    return opt;
}
