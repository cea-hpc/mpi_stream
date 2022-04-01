## CMake ##
cmake_minimum_required(VERSION 3.7)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -lm")
else()
   if("${CMAKE_C_COMPILER_ID}" STREQUAL "Fujitsu")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Kfast,openmp -Kprefetch_sequential=soft -Kzfill=18 -Kprefetch_line=9 -Kprefetch_line_L2=40")
   else()
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -qopt-streaming-stores=always")
   endif()
endif()

add_definitions(-D__MPI__)
add_executable(mpi_stream stream.c)
