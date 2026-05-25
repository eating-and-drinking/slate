// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// idx.c — IDX file format reader (MNIST native).

#include "slate/data_simple.h"
#include "slate/error.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

static slate_status_t load_idx(const char* path, slate_idx_data_t* out,
                                uint32_t expected_magic, int expected_dims) {
    if (!path || !out) return SLATE_ERR_INVALID_ARGUMENT;
    out->data = NULL;

    FILE* fp = fopen(path, "rb");
    if (!fp) return slate_set_error(SLATE_ERR_IO, "cannot open '%s'", path);

    uint8_t hdr[16];
    int hdr_len = (expected_dims == 3) ? 16 : 8;
    if (fread(hdr, 1, (size_t)hdr_len, fp) != (size_t)hdr_len) {
        fclose(fp);
        return slate_set_error(SLATE_ERR_IO, "short read on header of '%s'", path);
    }

    uint32_t magic = read_be32(hdr + 0);
    if (magic != expected_magic) {
        fclose(fp);
        return slate_set_error(SLATE_ERR_INVALID_ARGUMENT,
                               "bad IDX magic 0x%08x in '%s' (expected 0x%08x)",
                               magic, path, expected_magic);
    }

    out->n_items = (int)read_be32(hdr + 4);
    if (expected_dims == 3) {
        out->rows = (int)read_be32(hdr + 8);
        out->cols = (int)read_be32(hdr + 12);
    } else {
        out->rows = 1;
        out->cols = 1;
    }

    size_t total = (size_t)out->n_items * (size_t)out->rows * (size_t)out->cols;
    out->data = (uint8_t*)malloc(total);
    if (!out->data) { fclose(fp); return SLATE_ERR_OUT_OF_MEMORY; }

    if (fread(out->data, 1, total, fp) != total) {
        free(out->data);
        out->data = NULL;
        fclose(fp);
        return slate_set_error(SLATE_ERR_IO, "short read on payload of '%s'", path);
    }
    fclose(fp);
    return SLATE_OK;
}

slate_status_t slate_idx_load_images(const char* path, slate_idx_data_t* out) {
    return load_idx(path, out, 0x00000803u, 3);
}

slate_status_t slate_idx_load_labels(const char* path, slate_idx_data_t* out) {
    return load_idx(path, out, 0x00000801u, 1);
}

void slate_idx_free(slate_idx_data_t* d) {
    if (!d) return;
    free(d->data);
    d->data = NULL;
    d->n_items = d->rows = d->cols = 0;
}
