cmake_minimum_required(VERSION 3.1)

include_directories(../doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out pcas)

add_test(NAME doctest     COMMAND mpirun -n 1 doctest.out)
add_test(NAME doctest_np2 COMMAND mpirun -n 2 doctest.out)
add_test(NAME doctest_np4 COMMAND mpirun -n 4 doctest.out)
