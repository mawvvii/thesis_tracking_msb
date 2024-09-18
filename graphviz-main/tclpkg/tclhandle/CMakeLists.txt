add_library(tclhandle OBJECT
  tclhandle.c
  tclhandle.h
)

target_include_directories(tclhandle
  PRIVATE
    ../../lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(tclhandle SYSTEM PUBLIC ${TCL_INCLUDE_PATH})
target_link_libraries(tclhandle PUBLIC ${TCL_LIBRARY})
