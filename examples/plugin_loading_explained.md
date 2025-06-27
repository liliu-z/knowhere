# Plugin Loading Process Explained

## 1. Loading .so Files

```cpp
// Use dlopen to load dynamic library
void* handle = dlopen("plugin.so", RTLD_LAZY | RTLD_LOCAL);
```

- `RTLD_LAZY`: Lazy symbol resolution (resolve when needed)
- `RTLD_LOCAL`: Symbols won't be used by other dynamic libraries

## 2. Getting Export Functions

```cpp
// Get API version check function
auto get_version = (uint32_t(*)())dlsym(handle, "GetKnowherePluginAPIVersion");
uint32_t version = get_version();

// Get factory creation function
auto create_factory = (IPluginFactory*(*)())dlsym(handle, "CreateKnowherePluginFactory");
IPluginFactory* factory = create_factory();
```

## 3. Using Plugins

```cpp
// Create index through factory
auto index = factory->CreateIndex();

// Use index
index->Build(dataset, config);
auto result = index->Search(queries, config, bitset);
```

## 4. Key Points

### Symbol Export
Plugins must use `extern "C"` to export functions, avoiding C++ name mangling:

```cpp
extern "C" {
    uint32_t GetKnowherePluginAPIVersion() { return 1; }
    IPluginFactory* CreateKnowherePluginFactory() { return new MyFactory(); }
}
```

### Version Compatibility
Ensure interface compatibility through API version number:

```cpp
if (plugin_version != KNOWHERE_PLUGIN_API_VERSION) {
    // Version mismatch, refuse to load
    return Status::version_mismatch;
}
```

### Lifecycle Management
- Plugins can have load/unload hooks
- Use smart pointers for memory management
- Automatic resource cleanup on unload

## 5. Usage Examples

### Load All Plugins
```cpp
// Load all plugins from directory
PluginLoader::Instance().LoadPluginsFromDirectory("/path/to/plugins");
```

### Use Plugin Index
```cpp
// Plugins are automatically registered to IndexFactory
auto index = IndexFactory::Instance().Create("PLUGIN_AiSAQ");
```

### List Loaded Plugins
```cpp
auto plugins = PluginLoader::Instance().ListPlugins();
for (const auto& info : plugins) {
    std::cout << info.name << " v" << info.version << std::endl;
}
```

## 6. Security Considerations

1. **Path Validation**: Only load from trusted directories
2. **Signature Verification**: Can add digital signature verification
3. **Sandbox Isolation**: Can run in restricted environment
4. **Resource Limits**: Limit plugin resource usage

## 7. Cross-Platform Support

### Linux
```cpp
void* handle = dlopen("plugin.so", RTLD_LAZY);
```

### macOS
```cpp
void* handle = dlopen("plugin.dylib", RTLD_LAZY);
```

### Windows
```cpp
HMODULE handle = LoadLibrary("plugin.dll");
```

## 8. Error Handling

```cpp
if (!handle) {
    const char* error = dlerror();
    LOG_ERROR << "Failed to load plugin: " << error;
    // Common errors:
    // - File doesn't exist
    // - Architecture mismatch (32/64 bit)
    // - Missing dependencies
    // - Undefined symbols
}
```

## 9. Performance Considerations

- Loading is one-time, no runtime overhead
- Function calls through function pointers, performance close to direct calls
- Can preload frequently used plugins

## 10. Debugging Tips

```bash
# View exported symbols in .so file
nm -D plugin.so | grep " T "

# View dependencies
ldd plugin.so

# Set debug environment variable
export LD_DEBUG=all
```
