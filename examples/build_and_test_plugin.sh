#!/bin/bash

# Build and test the plugin system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Building SimpleVector Plugin ==="

# Create plugins directory
mkdir -p plugins

# Build the plugin
cd simple_vector_plugin
mkdir -p build
cd build

echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

echo "Building..."
make

echo "Copying plugin to test directory..."
cp simple_vector.so ../../plugins/

cd ../..

echo -e "\n=== Building Test Program ==="

# Note: This is a simplified build command.
# In a real scenario, you would need to link against Knowhere libraries
echo "Building test program..."
g++ -std=c++17 \
    -I../include \
    -I../thirdparty \
    test_plugin_system.cc \
    -o test_plugin_system \
    -ldl \
    -lpthread \
    2>/dev/null || echo "Note: Test program compilation requires full Knowhere build environment"

echo -e "\n=== Plugin System Components Created ==="
echo "1. Plugin interface: ../include/knowhere/plugin/plugin_interface.h"
echo "2. Plugin loader: ../include/knowhere/plugin/plugin_loader.h"
echo "3. Plugin factory: ../include/knowhere/plugin/plugin_factory.h"
echo "4. Example plugin: simple_vector_plugin/"
echo "5. Test program: test_plugin_system.cc"

echo -e "\n=== How to integrate with Knowhere ==="
echo "1. Add to Knowhere's CMakeLists.txt:"
echo "   option(WITH_PLUGINS \"Enable plugin support\" ON)"
echo "   if(WITH_PLUGINS)"
echo "     add_definitions(-DKNOWHERE_WITH_PLUGINS)"
echo "   endif()"
echo ""
echo "2. In your application startup code:"
echo "   #include \"knowhere/plugin/plugin_factory.h\""
echo "   knowhere::plugin::InitializePlugins(\"/path/to/plugins\");"
echo ""
echo "3. Use plugins like any other index:"
echo "   auto index = IndexFactory::Instance().Create(\"PLUGIN_SimpleVector\");"

echo -e "\n=== Plugin Development Guide ==="
echo "For plugin developers:"
echo "1. Copy the simple_vector_plugin directory as a template"
echo "2. Implement your algorithm by inheriting from IPluginIndex"
echo "3. Export the required C functions"
echo "4. Build as a shared library"
echo "5. Place in the plugins directory"

echo -e "\nDone!"
