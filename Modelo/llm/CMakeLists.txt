cmake_minimum_required(VERSION 3.14)
project(obgyn-chatbot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# === TODAS las DLLs y EXEs van al mismo directorio ===
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Build llama.cpp library (no tools/examples)
set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
set(LLAMA_AVX2 ON CACHE BOOL "" FORCE)
set(LLAMA_FMA ON CACHE BOOL "" FORCE)
set(LLAMA_F16C ON CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp ${CMAKE_BINARY_DIR}/llama-build)

# Our chatbot app (flags solo para nuestro target, no afectan a llama.cpp)
add_subdirectory(app)
