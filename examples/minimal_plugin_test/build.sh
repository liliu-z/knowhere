#!/bin/bash

echo "=== Compiling plugin ==="
# Compile to .so file
g++ -shared -fPIC -o plugin.so plugin.cc

echo "=== Compiling loader ==="
# Compile loader program
g++ -o loader loader.cc -ldl

echo "=== Running test ==="
./loader

echo "=== Plugin info ==="
echo "Plugin size:"
ls -lh plugin.so

echo "Exported symbols:"
nm -D plugin.so | grep " T "
