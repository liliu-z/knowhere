# Knowhere Plugin Development Guide

## Overview

The Knowhere Plugin System allows third-party developers to create custom vector index implementations without modifying Knowhere core code. This guide explains how to develop, build, and distribute your own plugins.

## Quick Start

### 1. Use the Template

Copy the example plugin as a starting point:

```bash
cp -r knowhere/examples/simple_vector_plugin my_awesome_index
cd my_awesome_index
```

### 2. Implement Your Index

Edit `src/simple_vector_index.h` to implement your algorithm:

```cpp
class MyAwesomeIndex : public plugin::IPluginIndex {
    // Implement required methods
    Status Build(const DataSet& dataset, const Config& cfg) override;
    expected<DataSetPtr> Search(...) override;
    // ... other methods
};
```

### 3. Build Your Plugin

```bash
mkdir build && cd build
cmake ..
make
```

### 4. Install and Use

```bash
# Copy to plugins directory
cp my_awesome.so /usr/local/lib/knowhere/plugins/

# In your application
knowhere::plugin::InitializePlugins();
auto index = IndexFactory::Instance().Create("PLUGIN_MyAwesome");
```

## Plugin Architecture

### Required Interfaces

Every plugin must implement:

1. **IPluginIndex** - Your index implementation
2. **IPluginConfig** - Configuration handling
3. **IPluginFactory** - Object creation
4. **Export Functions** - C-style plugin entry points

### Plugin Lifecycle

```
Load Plugin → Validate API Version → Create Factory → Register Index → Ready to Use
```

## Best Practices

### 1. Error Handling

Always return proper Status objects:

```cpp
if (data_invalid) {
    return Status::invalid_args("Data validation failed");
}
```

### 2. Thread Safety

Make your index thread-safe for concurrent searches:

```cpp
mutable std::shared_mutex mutex_;

Status Build(...) {
    std::unique_lock lock(mutex_);
    // Build implementation
}

expected<DataSetPtr> Search(...) const {
    std::shared_lock lock(mutex_);
    // Search implementation
}
```

### 3. Feature Declaration

Be accurate about supported features:

```cpp
PluginFeatures GetFeatures() const override {
    return {
        .supports_gpu = false,  // Set true only if you use CUDA
        .supports_mmap = true,  // Set true if you support memory mapping
        .supported_metrics = {"L2", "IP", "Cosine"}
    };
}
```

### 4. Configuration Validation

Always validate configuration parameters:

```cpp
Status Validate() const override {
    if (dim <= 0 || dim > 32768) {
        return Status::invalid_args("Invalid dimension");
    }
    return Status::success();
}
```

## Advanced Topics

### Custom Metrics

Implement custom distance functions:

```cpp
float CustomDistance(const float* a, const float* b, int dim) {
    // Your distance computation
}
```

### GPU Support

If your plugin uses GPU:

```cpp
#ifdef WITH_CUDA
class MyGPUIndex : public IPluginIndex {
    // GPU implementation
};
#endif
```

### Disk-Based Indexes

For disk-based indexes, accept FileManager:

```cpp
class MyDiskIndex : public IPluginIndex {
    std::shared_ptr<FileManager> file_manager_;

    Status LoadFromDisk(const std::string& path) {
        return file_manager_->LoadFile(path);
    }
};
```

## Distribution

### 1. Package Structure

```
my-awesome-index/
├── lib/
│   └── my_awesome.so
├── include/           # Optional: public headers
├── docs/
│   └── README.md
├── examples/
│   └── example.cc
└── LICENSE
```

### 2. Version Compatibility

Always specify compatible versions:

```yaml
# plugin.yaml
compatibility:
  knowhere_api: ["1.0"]
  milvus_version: ["2.3+"]
```

### 3. Documentation

Include:
- Algorithm description
- Performance characteristics
- Configuration parameters
- Usage examples
- Benchmarks

## Testing

### Unit Tests

```cpp
TEST(MyAwesomeIndex, BasicBuildSearch) {
    auto index = std::make_unique<MyAwesomeIndex>();
    // Test implementation
}
```

### Integration Tests

Test with Knowhere's test framework:

```cpp
INSTANTIATE_TEST_SUITE_P(
    MyAwesome,
    IndexTest,
    ::testing::Values(IndexTestParam{"PLUGIN_MyAwesome", /*...*/})
);
```

### Performance Benchmarks

Include benchmarks comparing to standard indexes:

```cpp
BENCHMARK(BM_MyAwesome_Build)->Range(1000, 1000000);
BENCHMARK(BM_MyAwesome_Search)->Range(1, 100);
```

## Debugging

### Enable Debug Logging

```cpp
LOG_KNOWHERE_DEBUG_ << "Processing vector " << i;
```

### Common Issues

1. **Plugin not loading**: Check API version and export functions
2. **Crashes**: Verify memory management and bounds checking
3. **Poor performance**: Profile and optimize hot paths

## Community

### Contributing Your Plugin

1. Publish to GitHub
2. Submit to knowhere-plugin-registry
3. Join the community discussions

### Getting Help

- GitHub Issues: Report bugs
- Discord: Real-time chat
- Mailing List: Design discussions

## Example Plugins

Study these examples:

1. **SimpleVector** - Basic brute-force search
2. **CompressedHNSW** - Advanced graph index with compression
3. **StreamingIVF** - Index with online updates
4. **SecureIndex** - Encrypted vector search

## FAQ

**Q: Can I use proprietary algorithms?**
A: Yes, plugins can be closed-source.

**Q: How do I handle updates?**
A: Implement the `OnUpgrade` lifecycle method.

**Q: Can plugins depend on external libraries?**
A: Yes, bundle dependencies or document installation.

**Q: How do I monetize my plugin?**
A: You can license your plugin commercially.

## Conclusion

The plugin system empowers you to extend Knowhere with custom algorithms while maintaining clean separation from core code. Start with the template, implement your algorithm, and share with the community!
