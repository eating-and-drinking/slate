set(SLATE_TESTS
    test_arena
    test_tensor
    test_autograd
    test_gradcheck
    test_transformer_block
    test_threadpool
    test_mha
    test_sampling
    test_bpe
    test_mmap_adafactor
    test_gguf
    test_quant
    test_gguf_q8_matmul
    test_q8_dot
    test_dpo
    test_kto_kd
    test_grpo
    test_sandbox
    test_checkpoint
    test_bf16_mode
    test_adapter_cache
    test_prm_http
    test_muon
    test_opd
    test_infer
    test_q4k
    test_gguf_q4k
    test_backend
    test_rope
    test_llama_load
    test_llama_infer
    test_llama_infer_q4k
    test_llama_infer_gqa
)
foreach(t ${SLATE_TESTS})
    add_executable(slate_${t} ${t}.c)
    target_link_libraries(slate_${t} PRIVATE slate)
    add_test(NAME ${t} COMMAND slate_${t})
endforeach()
