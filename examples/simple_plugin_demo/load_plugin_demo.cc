/**
 * Simple plugin loading demonstration
 * Shows how to dynamically load .so files and call functions within them
 */

#include <dlfcn.h>

#include <iostream>
#include <memory>

// Function types that plugins must implement
typedef const char* (*GetPluginNameFunc)();
typedef void* (*CreateIndexFunc)();
typedef void (*DestroyIndexFunc)(void*);

int
main() {
    // 1. Open .so file
    const char* plugin_path = "./simple_vector.so";
    void* handle = dlopen(plugin_path, RTLD_LAZY);

    if (!handle) {
        std::cerr << "Failed to load plugin: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded plugin: " << plugin_path << std::endl;

    // 2. Get functions from plugin
    // dlsym gets function pointer by function name string
    GetPluginNameFunc get_name = (GetPluginNameFunc)dlsym(handle, "GetPluginName");
    CreateIndexFunc create_index = (CreateIndexFunc)dlsym(handle, "CreateIndex");
    DestroyIndexFunc destroy_index = (DestroyIndexFunc)dlsym(handle, "DestroyIndex");

    if (!get_name || !create_index || !destroy_index) {
        std::cerr << "Plugin missing required export functions" << std::endl;
        dlclose(handle);
        return 1;
    }

    // 3. Call plugin functions
    const char* plugin_name = get_name();
    std::cout << "Plugin name: " << plugin_name << std::endl;

    // 4. Create index instance
    void* index = create_index();
    std::cout << "Created index instance: " << index << std::endl;

    // 5. Destroy when done
    destroy_index(index);
    std::cout << "Destroyed index instance" << std::endl;

    // 6. Close plugin
    dlclose(handle);
    std::cout << "Plugin unloaded" << std::endl;

    return 0;
}
