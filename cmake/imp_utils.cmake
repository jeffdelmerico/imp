# function(_imp_append_private_includes target)
#   if(DEFINED ${target}_PRIVATE_INCLUDE_DIRS)
#     target_include_directories(${target} PRIVATE ${target}_PRIVATE_INCLUDE_DIRS)
#     unset(${target}_PRIVATE_INCLUDE_DIRS CACHE)
#   endif()
# endfunction()

macro(imp_debug)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(STATUS "${__msg}")
endmacro()

macro(imp_warn)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(WARNING "${__msg}")
endmacro()

macro(imp_fatal)
  string(REPLACE ";" " " __msg "${ARGN}")
  message(FATAL_ERROR "${__msg}")
endmacro()
