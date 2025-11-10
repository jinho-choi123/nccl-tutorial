set(CSRC_DIR csrc)
file(GLOB_RECURSE CSRC_FILES ${CSRC_DIR}/*.cpp)
set(INCLUDE_DIR include)

add_executable(mt_md_main ${CSRC_FILES})
target_include_directories(mt_md_main PRIVATE ${INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(mt_md_main PRIVATE CUDA::cudart NCCL::NCCL ${MPI_CXX_LIBRARIES})
