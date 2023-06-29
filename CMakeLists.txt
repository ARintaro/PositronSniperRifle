
include_directories(include)
include_directories(lib)

file(GLOB_RECURSE HEADER_FILES *.h *.hpp)

file(GLOB_RECURSE SOURCE_FILES *.cpp *.c)

file(GLOB_RECURSE CUDA_FILES *.cu *.cuh)

set(AllFiles ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_FILES})

OPTIX_add_sample_executable(positronSniperRifle target_name
  ${AllFiles}
  OPTIONS -rdc true
)

target_link_libraries(${target_name}
  ${CUDA_LIBRARIES}
)
