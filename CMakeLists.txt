cmake_minimum_required(VERSION 3.10)
project(Cholesky C)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

add_executable(Cholesky main.c)