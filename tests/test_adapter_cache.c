#include "slate/adapter_mgr.h"
#include "slate/teacher_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(void) {
    int ok = 1;
    system("rm -rf /tmp/slate_amgr /tmp/slate_tcache.bin");

    // AdapterManager
    slate_adapter_mgr_t* m = slate_adapter_mgr_open("/tmp/slate_amgr");
    const char* v1 = "ADAPTER_VERSION_1_DATA";
    slate_adapter_mgr_write_candidate(m, v1, strlen(v1));
    slate_adapter_mgr_promote(m);
    printf("[amgr] after promote 1: archive_count=%d\n", slate_adapter_mgr_archive_count(m));
    const char* v2 = "ADAPTER_VERSION_2_DIFFERENT";
    slate_adapter_mgr_write_candidate(m, v2, strlen(v2));
    slate_adapter_mgr_promote(m);
    printf("[amgr] after promote 2: archive_count=%d (expect 1)\n", slate_adapter_mgr_archive_count(m));
    ok = ok && slate_adapter_mgr_archive_count(m) == 1;
    // Read current
    void* data; size_t sz;
    slate_adapter_mgr_read_current(m, &data, &sz);
    printf("[amgr] current = '%.*s'\n", (int)sz, (char*)data);
    ok = ok && memcmp(data, v2, strlen(v2)) == 0;
    free(data);
    slate_adapter_mgr_close(m);

    // TeacherCache
    slate_teacher_cache_t* c = slate_teacher_cache_open("/tmp/slate_tcache.bin");
    int32_t toks[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float lgs[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    slate_teacher_cache_put(c, 42, 2, 4, toks, lgs);
    slate_teacher_cache_put(c, 99, 1, 2, toks, lgs);
    printf("[tcache] size=%d\n", slate_teacher_cache_size(c));
    int sq, kk; int32_t* gt; float* gl;
    slate_teacher_cache_get(c, 42, &sq, &kk, &gt, &gl);
    printf("[tcache] get(42) seq=%d k=%d tokens[0..2]=%d %d %d  logit[3]=%.2f\n",
           sq, kk, gt[0], gt[1], gt[2], gl[3]);
    ok = ok && sq == 2 && kk == 4 && gt[0] == 0 && gl[3] == 0.4f;
    free(gt); free(gl);
    slate_teacher_cache_close(c);
    // Reopen and verify persistence
    c = slate_teacher_cache_open("/tmp/slate_tcache.bin");
    printf("[tcache] after reopen, size=%d\n", slate_teacher_cache_size(c));
    ok = ok && slate_teacher_cache_size(c) == 2;
    slate_teacher_cache_close(c);
    printf("test_adapter_cache: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
