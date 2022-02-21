cmake_minimum_required(VERSION 3.1)

# set(CMAKE_VERBOSE_MAKEFILE 1)

project(pcas VERSION 0.0.1 LANGUAGES CXX)

add_library(pcas INTERFACE)
target_include_directories(pcas INTERFACE include)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
