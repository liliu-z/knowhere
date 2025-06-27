// Simplest plugin implementation
#include <iostream>

extern "C" {
const char*
GetPluginName() {
    return "MinimalTestPlugin";
}

void*
CreateIndex() {
    std::cout << "[Plugin] Creating index instance" << std::endl;
    return new int(42);  // Return a simple object
}

void
DestroyIndex(void* index) {
    std::cout << "[Plugin] Destroying index instance" << std::endl;
    delete static_cast<int*>(index);
}
}
