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
)
foreach(t ${SLATE_TESTS})
    add_executable(slate_${t} ${t}.c)
    target_link_libraries(slate_${t} PRIVATE slate)
    add_test(NAME ${t} COMMAND slate_${t})
endforeach()
