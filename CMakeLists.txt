OPTIX_add_sample_executable(positronSniperRifle target_name
  optixLib.h
  cudaBuffer.h
  renderer.h
  renderParams.h
  window.h
  disney.h
  tiny_obj_loader.h
  shader.hpp
  devicePrograms.cu
  mesh.h
  mesh.cpp
  renderer.cpp
  main.cpp
  OPTIONS -rdc true
)

target_link_libraries(${target_name}
  ${CUDA_LIBRARIES}
)
