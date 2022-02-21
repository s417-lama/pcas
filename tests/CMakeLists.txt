cmake_minimum_required(VERSION 3.1)

include_directories(../doctest/doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out pcas)

add_test(NAME doctest COMMAND doctest.out)
