# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(nccl-tutorial)

#### Setup compiler ####

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#### Setup required dependencies ####
# Find NCCL
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
# Find NCCL
find_package(NCCL REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

#### Add source files ####
set(CSRC_DIR csrc)
file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)
set(INCLUDE_DIR include)


add_executable(main ${CSRC_FILES})
target_include_directories(main PRIVATE ${INCLUDE_DIR})
target_link_libraries(main PRIVATE CUDA::cudart NCCL::NCCL)
