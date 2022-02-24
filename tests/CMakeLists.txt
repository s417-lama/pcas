cmake_minimum_required(VERSION 3.1)

include_directories(../doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out pcas)

add_test(NAME doctest     COMMAND mpirun -n 1 doctest.out)
add_test(NAME doctest_np2 COMMAND mpirun -n 2 doctest.out)
add_test(NAME doctest_np4 COMMAND mpirun -n 4 doctest.out)

add_executable(matmul.out matmul.cpp)
target_link_libraries(matmul.out pcas)

add_test(NAME matmul     COMMAND mpirun -n 1 matmul.out)
add_test(NAME matmul_np2 COMMAND mpirun -n 2 matmul.out)
add_test(NAME matmul_np4 COMMAND mpirun -n 4 matmul.out)
