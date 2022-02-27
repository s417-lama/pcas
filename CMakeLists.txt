cmake_minimum_required(VERSION 3.13)

project(pcas VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

add_library(pcas INTERFACE)
target_include_directories(pcas INTERFACE include)
target_link_libraries(pcas INTERFACE ${MPI_CXX_LIBRARIES})
target_compile_options(pcas INTERFACE -Wall)

if(MPI_CXX_COMPILE_FLAGS)
  message(STATUS "MPI_CXX_COMPILE_FLAGS = ${MPI_CXX_COMPILE_FLAGS}")
  separate_arguments(MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND "${MPI_CXX_COMPILE_FLAGS}")
  target_compile_options(pcas INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
endif()
if(MPI_CXX_LINK_FLAGS)
  message(STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}")
  separate_arguments(MPI_CXX_LINK_OPTIONS UNIX_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(pcas INTERFACE ${MPI_CXX_LINK_OPTIONS})
endif()

include(CTest)
if(BUILD_TESTING)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()
