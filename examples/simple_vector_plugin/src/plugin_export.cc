/**
 * Plugin export implementation for SimpleVector
 */

#include "simple_vector_index.h"

using namespace knowhere::plugin;
using namespace simple_vector_plugin;

// Plugin factory implementation
class SimpleVectorFactory : public IPluginFactory {
 public:
    std::unique_ptr<IPluginIndex>
    CreateIndex() override {
        return std::make_unique<SimpleVectorIndex>();
    }

    std::unique_ptr<IPluginConfig>
    CreateConfig() override {
        return std::make_unique<SimpleVectorConfig>();
    }

    PluginInfo
    GetPluginInfo() const override {
        return PluginInfo{.name = "SimpleVector",
                          .version = "1.0.0",
                          .author = "Knowhere Example",
                          .description = "A simple brute-force vector search plugin",
                          .license = "MIT",
                          .api_version = KNOWHERE_PLUGIN_API_VERSION};
    }
};

// Plugin lifecycle implementation (optional)
class SimpleVectorLifecycle : public IPluginLifecycle {
 public:
    knowhere::Status
    OnLoad() override {
        LOG_KNOWHERE_INFO_ << "SimpleVector plugin loaded";
        return knowhere::Status::success();
    }

    knowhere::Status
    OnUnload() override {
        LOG_KNOWHERE_INFO_ << "SimpleVector plugin unloaded";
        return knowhere::Status::success();
    }

    knowhere::Status
    OnUpgrade(uint32_t from_version, uint32_t to_version) override {
        LOG_KNOWHERE_INFO_ << "SimpleVector plugin upgrade from " << from_version << " to " << to_version;
        return knowhere::Status::success();
    }
};

// C-style export functions
extern "C" {

uint32_t
GetKnowherePluginAPIVersion() {
    return KNOWHERE_PLUGIN_API_VERSION;
}

IPluginFactory*
CreateKnowherePluginFactory() {
    return new SimpleVectorFactory();
}

void
DestroyKnowherePluginFactory(IPluginFactory* factory) {
    delete factory;
}

IPluginLifecycle*
GetKnowherePluginLifecycle() {
    static SimpleVectorLifecycle lifecycle;
    return &lifecycle;
}

}  // extern "C"
