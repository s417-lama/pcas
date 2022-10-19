cmake_minimum_required(VERSION 3.13)

project(pcas VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

add_library(pcas INTERFACE)
target_include_directories(pcas INTERFACE include)
target_link_libraries(pcas INTERFACE ${MPI_CXX_LIBRARIES} rt)
target_compile_options(pcas INTERFACE -Wall -Wextra)

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

# FIXME: Hopefully remove Boost dependency in the future.
# This is only for platforms that do not support <memory_resource> of C++17.
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.60.0 COMPONENTS container)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries(pcas INTERFACE ${Boost_LIBRARIES})
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

include(CTest)
if(BUILD_TESTING)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()
