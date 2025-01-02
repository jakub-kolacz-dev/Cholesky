cmake_minimum_required(VERSION 3.10)
project(Cholesky C)
#project(Cholesky C CXX CUDA)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

find_package(MPI REQUIRED)
if(MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}")
endif()

add_executable(Cholesky main.c cholesky.cu
        cholesky.cu)
target_link_libraries(Cholesky m ${MPI_C_LIBRARIES})