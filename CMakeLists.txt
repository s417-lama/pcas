cmake_minimum_required(VERSION 3.1)

# set(CMAKE_VERBOSE_MAKEFILE 1)

project(pcas VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_library(pcas INTERFACE)
target_include_directories(pcas INTERFACE include)
target_link_libraries(pcas INTERFACE ${MPI_CXX_LIBRARIES})
target_compile_options(pcas INTERFACE -Wall)

include(CTest)
if(BUILD_TESTING)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()
