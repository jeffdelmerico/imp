project(imp_tools_cmd)
cmake_minimum_required(VERSION 2.8.0)

cmake_policy(SET CMP0054 OLD)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(imp_finders)
imp_setup()
imp_find_cuda()

###
### command line tools
###
# find_package( OpenCV REQUIRED core highgui imgproc)
# include_directories(${OpenCV_INCLUDE_DIRS})

cs_add_executable(image_pyramid_test image_pyramid_test.cpp)
target_link_libraries(image_pyramid_test)

cs_add_executable(image_filter_median3x3_test image_filter_median3x3_test.cpp)
target_link_libraries(image_filter_median3x3_test)

cs_add_executable(image_filter_gauss_test image_filter_gauss_test.cpp)
target_link_libraries(image_filter_gauss_test)

cs_add_executable(natural_edges_test natural_edges_test.cpp)
target_link_libraries(natural_edges_test)

cs_add_executable(cu_rof_denoising_test cu_rof_denoising_test.cpp)
target_link_libraries(cu_rof_denoising_test)

cs_add_executable(cu_tvl1_denoising_test cu_tvl1_denoising_test.cpp)
target_link_libraries(cu_tvl1_denoising_test)

cs_add_executable(pangolin_load_test pangolin_load_test.cpp)
target_link_libraries(pangolin_load_test)



#cs_add_executable(example_executable src/main.cc)
#target_link_libraries(example_executable example_library)

cs_install()
cs_export()
