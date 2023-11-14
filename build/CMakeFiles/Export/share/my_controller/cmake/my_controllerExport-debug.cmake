#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "my_controller::my_controller" for configuration "Debug"
set_property(TARGET my_controller::my_controller APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(my_controller::my_controller PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmy_controller.so"
  IMPORTED_SONAME_DEBUG "libmy_controller.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS my_controller::my_controller )
list(APPEND _IMPORT_CHECK_FILES_FOR_my_controller::my_controller "${_IMPORT_PREFIX}/lib/libmy_controller.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
