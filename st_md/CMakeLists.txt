set(CSRC_DIR csrc)
file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)
set(INCLUDE_DIR include)

add_executable(st_md_main ${CSRC_FILES})
target_include_directories(st_md_main PRIVATE ${INCLUDE_DIR})
target_link_libraries(st_md_main PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES})
