cmake_minimum_required(VERSION 3.5)
project(multi-kernels)


find_package(MPI REQUIRED)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")

if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -fdiagnostics-color=always")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

option(ENABLED_SYNERGY "Enable SYnergy support" OFF)
option(ENABLED_TIME_EVENT_PROFILING "Enable time profiling with sycl event" OFF)
option(WITH_MPI_ASYNCH "Enable MPI_Ixx call" OFF)
option(ENABLE_FREQ_CHANGE_MPI_HIDING "Enable frequency change MPI hiding" OFF)
option(WITH_PROCESS_FREQ_CHANGE "Enable freq. change with another MPI process" OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

if(DPCPP_WITH_CUDA_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

    set(CUDA_ARCH "" CACHE STRING "CUDA device architecture e.g. sm_70")

    if(NOT CUDA_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${CUDA_ARCH}")
    endif()
endif()

if(DPCPP_WITH_ROCM_BACKEND)
    set(ROCM_ARCH "" CACHE STRING "ROCm device architecture e.g. gfx908")

    if(NOT ROCM_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amd_gpu_${ROCM_ARCH}")
    endif()
endif()

if(DPCPP_WITH_LZ_BACKEND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_gen")

    set(LZ_ARCH "" CACHE STRING "Level Zero device architecture e.g. acm-g10")

    if(NOT LZ_ARCH STREQUAL "")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend \"-device ${LZ_ARCH}\"")
    endif()
endif()

set(benchmarks
    ace-sycl/ace_main.cpp
    aop-sycl/aop_main.cpp
    bh-sycl/bh_main.cpp
    srad-sycl/srad_main.cpp
    mnist-sycl/mnist_main.cpp
    metropolis-sycl/metropolis_main.cpp
    lulesh-sycl/lulesh_main.cc
    mpi-freq-change/mpi_app_geom_knn.cpp
    mpi-freq-change/mpi_app_geom_vec_add.cpp

    mpi-freq-change/freq_change_overhead.cpp

)

# Copy file
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/input.pgm
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/srad-sycl/input.pgm ${CMAKE_CURRENT_BINARY_DIR}/input.pgm
  DEPENDS ${CMAKE_SOURCE_DIR}/srad-sycl/input.pgm
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist-sycl/train-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte
  DEPENDS ${CMAKE_SOURCE_DIR}/mnist-sycl/train-images.idx3-ubyte
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist-sycl/train-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte
  DEPENDS ${CMAKE_SOURCE_DIR}/mnist-sycl/train-labels.idx1-ubyte
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/t10k-images.idx3-ubyte
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist-sycl/t10k-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-images.idx3-ubyte
  DEPENDS ${CMAKE_SOURCE_DIR}/mnist-sycl/t10k-images.idx3-ubyte
)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/t10k-labels.idx1-ubyte
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist-sycl/t10k-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-labels.idx1-ubyte
  DEPENDS ${CMAKE_SOURCE_DIR}/mnist-sycl/t10k-labels.idx1-ubyte
)

add_custom_target(copy_scripts ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/input.pgm ${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/t10k-labels.idx1-ubyte
)

# Setting variables
foreach(benchmark IN LISTS benchmarks)
    get_filename_component(target ${benchmark} NAME_WE)

    if (${target} STREQUAL "lulesh_main")
        add_executable(${target} ${benchmark} lulesh-sycl/lulesh-comm.cc lulesh-sycl/lulesh-init.cc lulesh-sycl/lulesh-util.cc lulesh-sycl/lulesh-viz.cc)
    else()
        add_executable(${target} ${benchmark})
    endif()

    if(ENABLED_SYNERGY)
        target_compile_definitions(${target} PUBLIC __ENABLED_SYNERGY)
        target_link_libraries(${target} PUBLIC synergy)
    endif()

    if(FREQ_SCALING)
        target_compile_definitions(${target} PUBLIC FREQ_SCALING)
    endif()

    if(ENABLED_TIME_EVENT_PROFILING)
        target_compile_definitions(${target} PUBLIC SYCL_BENCH_ENABLE_QUEUE_PROFILING=1)
    endif()
    if(WITH_MPI_ASYNCH)
        target_compile_definitions(${target} PUBLIC WITH_MPI_ASYNCH=1)
    else()
        target_compile_definitions(${target} PUBLIC WITH_MPI_ASYNCH=0)
    endif()

    if(ENABLE_FREQ_CHANGE_MPI_HIDING)
        target_compile_definitions(${target} PUBLIC HIDING=1)
    else()
        target_compile_definitions(${target} PUBLIC HIDING=0)
    endif()
   
    if(WITH_PROCESS_FREQ_CHANGE)
        target_compile_definitions(${target} PUBLIC WITH_PROCESS_FREQ_CHANGE=1)
    else()
        target_compile_definitions(${target} PUBLIC WITH_PROCESS_FREQ_CHANGE=0)
    endif()

    target_compile_definitions(${target} PRIVATE __DPCPP__)
    # Link MPI libraries
    target_link_libraries(${target} PUBLIC ${MPI_CXX_LIBRARIES})

    install(TARGETS ${target} RUNTIME DESTINATION bin/benchmarks/)
    get_filename_component(dir ${benchmark} DIRECTORY)
    set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(benchmark)

# The "compiletime" target should only be used in the context of the compile time evaluation script
# set_target_properties(compiletime PROPERTIES EXCLUDE_FROM_ALL 1)
if(ENABLED_SYNERGY)
    add_subdirectory(SYnergy)
endif()

# install(PROGRAMS bin/run-suite DESTINATION bin/)