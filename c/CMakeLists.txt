cmake_minimum_required(VERSION 3.7)

# Set the project name
project(mpi_stream VERSION 0.3)

# Set options
option(MPI "Enable MPI" OFF)
option(OPENMP "Enable OpenMP" OFF)

# Check if sqrt is present
include(CheckSymbolExists)
check_symbol_exists(sqrt "math.h" HAVE_SQRT)
if(NOT (HAVE_SQRT))
  unset(HAVE_SQRT CACHE)
  set(CMAKE_REQUIRED_LIBRARIES "m")
  check_symbol_exists(sqrt "math.h" HAVE_SQRT)
endif()

# Set default C compiler options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -g")

# MPI
if(MPI)
  find_package(MPI REQUIRED COMPONENTS C)
endif(MPI)

if(MPI_FOUND)
  set(HAVE_MPI ON)
  set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
endif(MPI_FOUND)

# OpenMP
if(OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif(OPENMP)

# Generate config.h
include_directories("${PROJECT_BINARY_DIR}")
configure_file(config_cmake.h.in config.h)

# Use src directory
add_subdirectory(src)


