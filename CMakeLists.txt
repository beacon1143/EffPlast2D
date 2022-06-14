cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(EffPlast2D CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(EffPlast2D 
  main.cu 
  EffPlast2D.h EffPlast2D.cu 
)

#set_target_properties(EffPlast2D PROPERTIES CUDA_ARCHITECTURES "70;75")
set_property(TARGET EffPlast2D PROPERTY CUDA_SEPARABLE_COMPILATION ON)
