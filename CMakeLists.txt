# =============================================================================
# Slate — top-level CMake build
# =============================================================================
cmake_minimum_required(VERSION 3.16)

project(slate
    VERSION 0.1.0
    DESCRIPTION "A pure C/C++ framework for training and fine-tuning language models on consumer hardware"
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(SLATE_BUILD_TESTS    "Build the unit-test suite"      ON)
option(SLATE_BUILD_EXAMPLES "Build the example programs"     ON)
option(SLATE_ENABLE_AVX2    "Enable AVX2 kernels on x86_64"  ON)
option(SLATE_ENABLE_NEON    "Enable NEON kernels on aarch64" ON)
option(SLATE_ENABLE_OPENMP  "Use OpenMP instead of the built-in threadpool" OFF)
set(SLATE_SANITIZE ""    CACHE STRING "Sanitizer to enable: address, undefined, thread, or empty")

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wstrict-prototypes
        -Wmissing-prototypes
        -fno-omit-frame-pointer
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3)
    endif()
endif()

# Sanitizers
if(SLATE_SANITIZE)
    message(STATUS "Slate: enabling sanitizer '${SLATE_SANITIZE}'")
    add_compile_options(-fsanitize=${SLATE_SANITIZE} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${SLATE_SANITIZE})
endif()

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(SLATE_ARCH_X86_64 ON)
    if(SLATE_ENABLE_AVX2 AND NOT MSVC)
        add_compile_options(-mavx2 -mfma)
    elseif(SLATE_ENABLE_AVX2 AND MSVC)
        add_compile_options(/arch:AVX2)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64)")
    set(SLATE_ARCH_AARCH64 ON)
endif()

# -----------------------------------------------------------------------------
# Threading
# -----------------------------------------------------------------------------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(slate STATIC
    src/runtime/arena.c
    src/runtime/threadpool.c
    src/runtime/log.c
    src/runtime/error.c
    src/runtime/pool_global.c
    src/tensor/tensor.c
    src/autograd/graph.c
    src/ops/matmul.c
    src/ops/elementwise.c
    src/ops/activation.c
    src/ops/loss.c
    src/ops/softmax.c
    src/ops/cross_entropy.c
    src/ops/add_bias.c
    src/ops/silu.c
    src/ops/scale.c
    src/ops/transpose.c
    src/ops/embedding.c
    src/ops/rms_norm.c
    src/ops/bmm.c
    src/ops/causal_mask.c
    src/ops/linear3d.c
    src/ops/permute12.c
    src/module/module.c
    src/module/linear.c
    src/module/sequential.c
    src/module/embedding.c
    src/module/rmsnorm.c
    src/module/attention.c
    src/module/mh_attention.c
    src/module/feedforward.c
    src/module/transformer_block.c
    src/module/causal_lm.c
    src/optim/sgd.c
    src/optim/adamw.c
    src/optim/lr_scheduler.c
    src/data/idx.c
    src/data/simple_dataloader.c
    src/tokenizer/char_tokenizer.c
    src/util/sampling.c
    src/tokenizer/bpe.c
    src/data/mmap_dataset.c
    src/optim/adafactor.c
)

target_include_directories(slate
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(slate PUBLIC Threads::Threads)

if(NOT WIN32)
    target_link_libraries(slate PUBLIC m)
endif()

set_target_properties(slate PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
)

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
if(SLATE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SLATE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Install rules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS slate
    EXPORT slateTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Slate ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler      : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Target arch     : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Tests           : ${SLATE_BUILD_TESTS}")
message(STATUS "  Examples        : ${SLATE_BUILD_EXAMPLES}")
message(STATUS "  Sanitizer       : '${SLATE_SANITIZE}'")
message(STATUS "")
