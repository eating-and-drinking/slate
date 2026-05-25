// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
#define _POSIX_C_SOURCE 200809L
#include "slate/adapter_mgr.h"
#include "slate/error.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct slate_adapter_mgr {
    char* root;
    char* current_path;
    char* tmp_path;
    char* archive_dir;
};

static char* join(const char* a, const char* b) {
    size_t n = strlen(a) + 1 + strlen(b) + 1;
    char* p = (char*)malloc(n);
    snprintf(p, n, "%s/%s", a, b);
    return p;
}
static int mkdir_p(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(path, 0755);
}

slate_adapter_mgr_t* slate_adapter_mgr_open(const char* root) {
    if (!root) return NULL;
    if (mkdir_p(root) != 0) return NULL;
    slate_adapter_mgr_t* m = (slate_adapter_mgr_t*)calloc(1, sizeof(*m));
    m->root = strdup(root);
    m->current_path = join(root, "current.lora");
    m->tmp_path = join(root, "training.lora.tmp");
    m->archive_dir = join(root, "archive");
    mkdir_p(m->archive_dir);
    return m;
}
void slate_adapter_mgr_close(slate_adapter_mgr_t* m) {
    if (!m) return;
    free(m->root); free(m->current_path); free(m->tmp_path); free(m->archive_dir);
    free(m);
}
slate_status_t slate_adapter_mgr_write_candidate(slate_adapter_mgr_t* m,
                                                  const void* data, size_t size) {
    if (!m || !data) return SLATE_ERR_INVALID_ARGUMENT;
    char* w = join(m->root, "training.lora.writing");
    FILE* fp = fopen(w, "wb"); if (!fp) { free(w); return SLATE_ERR_IO; }
    if (fwrite(data, 1, size, fp) != size) { fclose(fp); free(w); return SLATE_ERR_IO; }
    fflush(fp); fsync(fileno(fp)); fclose(fp);
    if (rename(w, m->tmp_path) != 0) { free(w); return SLATE_ERR_IO; }
    free(w); return SLATE_OK;
}
slate_status_t slate_adapter_mgr_promote(slate_adapter_mgr_t* m) {
    if (!m) return SLATE_ERR_INVALID_ARGUMENT;
    struct stat st;
    if (stat(m->current_path, &st) == 0) {
        char name[64]; time_t t = time(NULL); struct tm tm; gmtime_r(&t, &tm);
        snprintf(name, sizeof(name), "%04d-%02d-%02d-%02d%02d%02d.lora",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                 tm.tm_hour, tm.tm_min, tm.tm_sec);
        char* arch = join(m->archive_dir, name);
        // If the same-named archive already exists (same second), make it unique
        struct stat ast;
        if (stat(arch, &ast) == 0) {
            char name2[80]; snprintf(name2, sizeof(name2), "%s.dup", name);
            free(arch); arch = join(m->archive_dir, name2);
        }
        if (rename(m->current_path, arch) != 0) { free(arch); return SLATE_ERR_IO; }
        free(arch);
    }
    if (rename(m->tmp_path, m->current_path) != 0) return SLATE_ERR_IO;
    return SLATE_OK;
}
slate_status_t slate_adapter_mgr_rollback(slate_adapter_mgr_t* m, const char* name) {
    if (!m || !name) return SLATE_ERR_INVALID_ARGUMENT;
    char* src = join(m->archive_dir, name);
    struct stat st;
    if (stat(src, &st) != 0) { free(src); return SLATE_ERR_IO; }
    if (rename(src, m->current_path) != 0) { free(src); return SLATE_ERR_IO; }
    free(src); return SLATE_OK;
}
slate_status_t slate_adapter_mgr_read_current(slate_adapter_mgr_t* m,
                                               void** out_data, size_t* out_size) {
    if (!m || !out_data || !out_size) return SLATE_ERR_INVALID_ARGUMENT;
    FILE* fp = fopen(m->current_path, "rb"); if (!fp) return SLATE_ERR_IO;
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    void* buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return SLATE_ERR_IO; }
    fclose(fp);
    *out_data = buf; *out_size = (size_t)sz; return SLATE_OK;
}
int slate_adapter_mgr_archive_count(slate_adapter_mgr_t* m) {
    if (!m) return 0;
    DIR* d = opendir(m->archive_dir); if (!d) return 0;
    int count = 0; struct dirent* e;
    while ((e = readdir(d))) if (e->d_name[0] != '.' && strstr(e->d_name, ".lora")) count++;
    closedir(d); return count;
}
