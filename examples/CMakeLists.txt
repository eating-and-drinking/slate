add_executable(slate_xor 01_xor/main.c)
target_link_libraries(slate_xor PRIVATE slate)

add_executable(slate_synth_cls 02_synth_cls/main.c)
target_link_libraries(slate_synth_cls PRIVATE slate)

add_executable(slate_mnist 02_mnist/main.c)
target_link_libraries(slate_mnist PRIVATE slate)

add_executable(slate_synth_seq 03_synth_seq/main.c)
target_link_libraries(slate_synth_seq PRIVATE slate)

add_executable(slate_tinyshakespeare 03_tinyshakespeare/main.c)
target_link_libraries(slate_tinyshakespeare PRIVATE slate)

add_executable(slate_gpt2 04_gpt2/main.c)
target_link_libraries(slate_gpt2 PRIVATE slate)

add_executable(slate_opd 10_opd/main.c)
target_link_libraries(slate_opd PRIVATE slate)
