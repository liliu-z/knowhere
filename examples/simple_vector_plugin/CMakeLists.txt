cmake_minimum_required(VERSION 3.18)
project(simple_vector_plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Knowhere headers
# In a real plugin, this would be installed or provided
set(KNOWHERE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include" CACHE PATH "Path to Knowhere headers")

# Plugin sources
set(PLUGIN_SOURCES
    src/plugin_export.cc
)

# Create shared library
add_library(simple_vector_plugin SHARED ${PLUGIN_SOURCES})

# Include directories
target_include_directories(simple_vector_plugin PRIVATE
    ${KNOWHERE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compile options
target_compile_options(simple_vector_plugin PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Link options
target_link_options(simple_vector_plugin PRIVATE
    -Wl,-z,defs
)

# Set output name without lib prefix
set_target_properties(simple_vector_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "simple_vector"
)

# Installation rules
install(TARGETS simple_vector_plugin
    LIBRARY DESTINATION lib/knowhere/plugins
)
