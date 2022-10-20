cmake_minimum_required(VERSION 3.1)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

include_directories(../doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out pcas)

add_test(NAME doctest     COMMAND ${MPIEXEC} -n 1 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np2 COMMAND ${MPIEXEC} -n 2 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np4 COMMAND ${MPIEXEC} -n 4 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)

set_tests_properties(doctest doctest_np2 doctest_np4 PROPERTIES ENVIRONMENT "PCAS_ENABLE_SHARED_MEMORY=0")

add_executable(matmul.out matmul.cpp)
target_link_libraries(matmul.out pcas)

add_test(NAME matmul     COMMAND ${MPIEXEC} -n 1 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./matmul.out)
add_test(NAME matmul_np2 COMMAND ${MPIEXEC} -n 2 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./matmul.out)
add_test(NAME matmul_np4 COMMAND ${MPIEXEC} -n 4 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./matmul.out)

set_tests_properties(matmul matmul_np2 matmul_np4 PROPERTIES ENVIRONMENT "PCAS_ENABLE_SHARED_MEMORY=0")
