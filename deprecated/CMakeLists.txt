set(PROJECT_NAME imp)
#set(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
#string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_VAR)
project(${PROJECT_NAME} CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8.0)

## base directory
if(NOT IMP_PATH)
  set(IMP_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# ## internal macros/includes
# include(${IMP_PATH}/cmake/imp_utils.cmake)
# include(${IMP_PATH}/cmake/imp_module.cmake)
# include(${IMP_PATH}/cmake/imp_ext_finders.cmake)


# # cleanup
# set(IMP_MODULE_LIBRARIES_LOCATIONS "" CACHE INTERNAL "List of the absolute locations of all built module libraries")
# set(IMP_MODULE_INCLUDE_PATHS "" CACHE INTERNAL "List of all module's include paths")
# set(IMP_LINK_DEPS "" CACHE INTERNAL "linkage dependencies for imp")

# #mark_as_advanced(IMP_MODULE_LIBRARIES_LOCATIONS IMP_MODULE_INCLUDE_PATHS)

# # set(${IU_PUBLIC_HEADERS} "")

# # # general settings
# # set( CMAKE_DEBUG_POSTFIX "d" )

# # #FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
# # FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
# # SET(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})

# # convert env variable to internal path to circumvent win32 backslash problems
# #FILE(TO_CMAKE_PATH "$ENV{VMLIBRARIES_ROOT}" FL_VMLIBRARIES_ROOT)

# ##-----------------------------------------------------------------------------
# ## Current version number:
# ##-----------------------------------------------------------------------------
# set(${PROJECT_NAME}_VERSION "2.0.0")

# string(REGEX MATCHALL "[0-9]" ${PROJECT_NAME}_VERSION_PARTS "${${PROJECT_NAME}_VERSION}")

# list(GET ${PROJECT_NAME}_VERSION_PARTS 0 ${PROJECT_NAME}_VERSION_MAJOR)
# list(GET ${PROJECT_NAME}_VERSION_PARTS 1 ${PROJECT_NAME}_VERSION_MINOR)
# list(GET ${PROJECT_NAME}_VERSION_PARTS 2 ${PROJECT_NAME}_VERSION_PATCH)
# set(${PROJECT_NAME}_SOVERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}")

# if(WIN32)
#    # Postfix of DLLs:
#     set(${PROJECT_NAME}_DLLVERSION "${${PROJECT_NAME}_VERSION_MAJOR}${${PROJECT_NAME}_VERSION_MINOR}${${PROJECT_NAME}_VERSION_PATCH}")
#     set(${PROJECT_NAME}_DEBUG_POSTFIX d)
# else()
#     # Postfix of so's:
#     #set(${PROJECT_NAME}_DLLVERSION "${${PROJECT_NAME}_VERSION_MAJOR}${${PROJECT_NAME}_VERSION_MINOR}${${PROJECT_NAME}_VERSION_PATCH}")
#     set(${PROJECT_NAME}_DLLVERSION "")
#     set(${PROJECT_NAME}_DEBUG_POSTFIX)
# endif()

# # ##-----------------------------------------------------------------------------
# # ## Flags if modules are used.
# # ## The modules are included per default unless dependencies not found or omitted by the user.
# # ## (the core module is always included!)
# # OPTION(${PROJECT_NAME}_USE_SPARSE "Including Sparse Matrix module." OFF)
# # OPTION(${PROJECT_NAME}_USE_IPP "Including IPP module." ON)
# # OPTION(${PROJECT_NAME}_USE_MATLAB "Including Matlab module." ON)
# # OPTION(${PROJECT_NAME}_USE_GUI "Including GUI module." OFF)
# # OPTION(${PROJECT_NAME}_USE_IO "Including IO module." ON)
# # OPTION(${PROJECT_NAME}_USE_IOPGM "Including IOPGM module." ON)
# # #OPTION(${PROJECT_NAME}_USE_VIDEOCAPTURE "Including VideCapture IO module." ON)
# # #OPTION(${PROJECT_NAME}_USE_PGRCAMERA "Including PointGray IO module." ON)
# # OPTION(${PROJECT_NAME}_USE_VIDEOSOURCE "Including Videosource IO module." ON)


# ##-----------------------------------------------------------------------------
# ## Compiler Flags
# if(UNIX)
#   #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x") # come on guys... update your system ;)
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#   add_definitions(-Wall)
#   add_definitions(-Werror)
#   add_definitions(-g)
#   add_definitions(-fPIC)
# endif()

# IF(WIN32)
#   ADD_DEFINITIONS(-DWIN32)
#   # for msvc 8 deprecated warning removal
#   ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
#   ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
#   ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)

#   SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
# endif(WIN32)

# ##-----------------------------------------------------------------------------
# ## settings / options
# option(IMP_BUILD_SHARED_LIBS "Build shared libraries if ON (default) or static ones if OFF." ON)
# option(IMP_THROW_ON_CUDA_ERROR "Throws an exception when cuda errors are checked with the internal macro." ON)
# if (${IMP_THROW_ON_CUDA_ERROR})
#    add_definitions(-DIMP_THROW_ON_CUDA_ERROR)
# endif()

# ##-----------------------------------------------------------------------------
# ## 3rd party dependencies
# option(IMP_WITH_OPENCV "Build IMP with opencv connectors. Default is ON." ON)
# option(IMP_WITH_CUDA "Build IMP with CUDA support. Default is OFF." ON)

# ##-----------------------------------------------------------------------------
# ## include modules
# add_subdirectory(modules)

# ##-----------------------------------------------------------------------------
# ## print build config
# ##
# imp_debug("\n\n***********************************************************************************")
# imp_debug("IMP BUILD SETUP:")
# imp_debug("================")
# imp_debug("")
# imp_debug("IMP_PATH:" ${IMP_PATH})
# imp_debug("IMP_BUILD_SHARED_LIBS:" ${IMP_BUILD_SHARED_LIBS})
# imp_debug("IMP_WITH_OPENCV:" ${IMP_WITH_OPENCV})
# imp_debug("IMP_WITH_CUDA:" ${IMP_WITH_CUDA})
# imp_debug("------------------------------------------------------------")
# imp_debug("IMP_MODULE_LIBRARIES_LOCATIONS: ${IMP_MODULE_LIBRARIES_LOCATIONS}")
# imp_debug("IMP_MODULE_INCLUDE_PATHS: ${IMP_MODULE_INCLUDE_PATHS}")
# imp_debug("\n***********************************************************************************\n")

# #
# #

# ##-----------------------------------------------------------------------------
# ## impConfig.cmake creation for finding IMP
# # Create the opengvConfig.cmake file for other cmake projects.
# #get_target_property(FULL_LIBRARY_NAME ${PROJECT_NAME} LOCATION)
# set(IMP_LIBRARIES ${IMP_MODULE_LIBRARIES_LOCATIONS})
# message(STATUS "link_deps ${IMP_core_LINK_DEPS}")
# set(IMP_LIBRARY_DEPENDS ${IMP_LINK_DEPS})
# set(IMP_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
# set(IMP_INCLUDE_DIR ${IMP_MODULE_INCLUDE_PATHS})
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/impConfig.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/impConfig.cmake @ONLY IMMEDIATE)
# export(PACKAGE ${PROJECT_NAME})


# ##-----------------------------------------------------------------------------
# ## add tests
# add_subdirectory(tests)
