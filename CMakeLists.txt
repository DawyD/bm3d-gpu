cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(bm3d LANGUAGES CXX CUDA)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# The following lines build the main executable.
#add_executable(bm3d
#	include/CImg.h
#	include/stopwatch.hpp
#	include/indices.cuh
#	include/params.hpp
#	include/bm3d.hpp
#	src/filtering.cu
#	src/blockmatching.cu
#	src/dct8x8.cu
#	src/main_nodisplay.cpp
#)

add_library(bm3d
	include/CImg.h
	include/stopwatch.hpp
	include/indices.cuh
	include/params.hpp
	include/bm3d.hpp
	src/filtering.cu
	src/blockmatching.cu
	src/dct8x8.cu
)

target_link_libraries(bm3d cufft cudart png)
