/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_node.h"

#define KNOWHERE_PLUGIN_API_VERSION 1

namespace knowhere::plugin {

// Plugin metadata information
struct PluginInfo {
    const char* name;
    const char* version;
    const char* author;
    const char* description;
    const char* license;
    uint32_t api_version;
};

// Plugin status for health checks
enum class PluginStatus { HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN };

// Plugin feature declarations
struct PluginFeatures {
    bool supports_gpu = false;
    bool supports_mmap = false;
    bool supports_range_search = false;
    bool supports_iterator = false;
    bool supports_disk_storage = false;
    std::vector<std::string> supported_metrics;
    std::vector<std::string> supported_data_types;
};

// Base plugin index interface
class IPluginIndex : public IndexNode {
 public:
    // Plugin metadata
    virtual PluginInfo
    GetPluginInfo() const = 0;

    // Health check
    virtual PluginStatus
    HealthCheck() const = 0;

    // Feature declarations
    virtual PluginFeatures
    GetFeatures() const = 0;

    // Optional: Get plugin-specific metrics
    virtual expected<Json>
    GetMetrics() const {
        return expected<Json>::Err(Status::not_implemented, "metrics not implemented");
    }
};

// Plugin configuration base class
class IPluginConfig : public BaseConfig {
 public:
    virtual ~IPluginConfig() = default;

    // Validate configuration
    virtual Status
    Validate() const = 0;

    // Get default configuration
    virtual Json
    GetDefaultConfig() const = 0;
};

// Plugin factory interface
class IPluginFactory {
 public:
    virtual ~IPluginFactory() = default;

    // Create index instance
    virtual std::unique_ptr<IPluginIndex>
    CreateIndex() = 0;

    // Create configuration instance
    virtual std::unique_ptr<IPluginConfig>
    CreateConfig() = 0;

    // Get plugin info
    virtual PluginInfo
    GetPluginInfo() const = 0;
};

// Plugin lifecycle hooks (optional)
class IPluginLifecycle {
 public:
    virtual ~IPluginLifecycle() = default;

    // Called when plugin is loaded
    virtual Status
    OnLoad() {
        return Status::success();
    }

    // Called when plugin is unloaded
    virtual Status
    OnUnload() {
        return Status::success();
    }

    // Called to upgrade plugin data
    virtual Status
    OnUpgrade(uint32_t from_version, uint32_t to_version) {
        return Status::success();
    }
};

}  // namespace knowhere::plugin

// C-style export functions that plugins must implement
extern "C" {
// Get plugin API version
uint32_t
GetKnowherePluginAPIVersion();

// Create plugin factory
knowhere::plugin::IPluginFactory*
CreateKnowherePluginFactory();

// Destroy plugin factory
void
DestroyKnowherePluginFactory(knowhere::plugin::IPluginFactory* factory);

// Optional: Get lifecycle handler
knowhere::plugin::IPluginLifecycle*
GetKnowherePluginLifecycle();
}
