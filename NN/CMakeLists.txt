cmake_minimum_required(VERSION 3.16)
project(dodnn LANGUAGES CXX)

# ---------------------------------------------------------------------------
# dodnn: a small static library with two independent models.
#   FNN  : FNN.h  FNN_io.h  FNN.cpp
#   LSTM : LSTM.h LSTM_io.h LSTM.cpp
# They share only functions.h / io_common.h (tiny, header-only).
# Turn a model OFF and its .cpp is not compiled, not archived, not linked,
# and including its header becomes a compile error (see DOD_NO_FNN/DOD_NO_LSTM).
# ---------------------------------------------------------------------------
option(DODNN_ENABLE_FNN  "Compile the feed-forward network" ON)
option(DODNN_ENABLE_LSTM "Compile the LSTM"                 ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_dodnn_top ON)
else()
  set(_dodnn_top OFF)          # used via add_subdirectory(): don't build demos/tests by default
endif()
option(DODNN_BUILD_EXAMPLES "Build the demo programs" ${_dodnn_top})
option(DODNN_BUILD_TESTS    "Build and register the tests" ${_dodnn_top})

if(NOT DODNN_ENABLE_FNN AND NOT DODNN_ENABLE_LSTM)
  message(FATAL_ERROR "dodnn: enable at least one of DODNN_ENABLE_FNN / DODNN_ENABLE_LSTM")
endif()

set(DODNN_SOURCES "")
if(DODNN_ENABLE_FNN)
  list(APPEND DODNN_SOURCES FNN.cpp)
endif()
if(DODNN_ENABLE_LSTM)
  list(APPEND DODNN_SOURCES LSTM.cpp)
endif()

add_library(dodnn STATIC ${DODNN_SOURCES})
target_include_directories(dodnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(dodnn PUBLIC cxx_std_17)

# Consumers inherit these, so `#include "LSTM.h"` fails loudly if LSTM was left out.
if(NOT DODNN_ENABLE_FNN)
  target_compile_definitions(dodnn PUBLIC DOD_NO_FNN)
endif()
if(NOT DODNN_ENABLE_LSTM)
  target_compile_definitions(dodnn PUBLIC DOD_NO_LSTM)
endif()

if(MSVC)
  target_compile_options(dodnn PRIVATE /W4)
else()
  target_compile_options(dodnn PRIVATE -Wall -Wextra)
endif()

# ------------------------------------------------------------------ examples
if(DODNN_BUILD_EXAMPLES)
  if(DODNN_ENABLE_FNN)
    add_executable(fnn_demo main.cpp)
    target_link_libraries(fnn_demo PRIVATE dodnn)
  endif()
  if(DODNN_ENABLE_LSTM)
    add_executable(lstm_demo examples/lstm_demo.cpp)
    target_link_libraries(lstm_demo PRIVATE dodnn)
  endif()
endif()

# --------------------------------------------------------------------- tests
if(DODNN_BUILD_TESTS)
  enable_testing()
  if(DODNN_ENABLE_FNN)
    add_executable(fnn_test tests/fnn_test.cpp)
    target_link_libraries(fnn_test PRIVATE dodnn)
    add_test(NAME fnn_test COMMAND fnn_test)
  endif()
  if(DODNN_ENABLE_LSTM)
    add_executable(lstm_test tests/lstm_test.cpp)
    target_link_libraries(lstm_test PRIVATE dodnn)
    add_test(NAME lstm_test COMMAND lstm_test)
  endif()
endif()
