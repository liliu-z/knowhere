// Loader program
#include <dlfcn.h>

#include <iostream>

int
main() {
    std::cout << "=== Plugin Loading Test ===" << std::endl;

    // 1. Load plugin
    void* handle = dlopen("./plugin.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Load failed: " << dlerror() << std::endl;
        return 1;
    }

    // 2. Get functions
    auto get_name = (const char* (*)())dlsym(handle, "GetPluginName");
    auto create = (void* (*)())dlsym(handle, "CreateIndex");
    auto destroy = (void (*)(void*))dlsym(handle, "DestroyIndex");

    // 3. Use plugin
    std::cout << "Plugin name: " << get_name() << std::endl;
    void* index = create();
    destroy(index);

    // 4. Unload
    dlclose(handle);
    std::cout << "Test complete!" << std::endl;

    return 0;
}
