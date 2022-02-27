cmake_minimum_required(VERSION 3.1)

include_directories(../doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out pcas)

add_test(NAME doctest     COMMAND ${MPIEXEC} -n 1 doctest.out)
add_test(NAME doctest_np2 COMMAND ${MPIEXEC} -n 2 doctest.out)
add_test(NAME doctest_np4 COMMAND ${MPIEXEC} -n 4 doctest.out)

add_executable(matmul.out matmul.cpp)
target_link_libraries(matmul.out pcas)

add_test(NAME matmul     COMMAND ${MPIEXEC} -n 1 matmul.out)
add_test(NAME matmul_np2 COMMAND ${MPIEXEC} -n 2 matmul.out)
add_test(NAME matmul_np4 COMMAND ${MPIEXEC} -n 4 matmul.out)
