/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "knowhere/factory.h"
#include "knowhere/plugin/plugin_interface.h"
#include "knowhere/plugin/plugin_loader.h"

namespace knowhere::plugin {

// Plugin-aware factory extension
class PluginFactory {
 public:
    static PluginFactory&
    Instance() {
        static PluginFactory instance;
        return instance;
    }

    // Register a plugin with the main IndexFactory
    Status
    RegisterPlugin(const std::string& name) {
        auto* plugin_factory = PluginLoader::Instance().GetPluginFactory(name);
        if (!plugin_factory) {
            return Status::invalid_args_in_format("Plugin {} not loaded", name);
        }

        // Get plugin info
        auto info = plugin_factory->GetPluginInfo();

        // Create a wrapper function that creates the plugin index
        auto create_func = [plugin_factory]() -> expected<std::unique_ptr<IndexNode>> {
            auto index = plugin_factory->CreateIndex();
            if (!index) {
                return expected<std::unique_ptr<IndexNode>>::Err(Status::invalid_args("Failed to create plugin index"),
                                                                 nullptr);
            }
            return std::unique_ptr<IndexNode>(std::move(index));
        };

        // Register with the main IndexFactory
        // Use a special prefix for plugin indexes
        std::string full_name = "PLUGIN_" + std::string(info.name);

        // Get plugin features
        auto temp_index = plugin_factory->CreateIndex();
        auto features = temp_index->GetFeatures();

        // Build feature set
        std::set<IndexFactory::Feature> feature_set;
        if (features.supports_gpu) {
            feature_set.insert(IndexFactory::Feature::GPU);
        }
        if (features.supports_mmap) {
            feature_set.insert(IndexFactory::Feature::MMAP);
        }
        if (features.supports_disk_storage) {
            feature_set.insert(IndexFactory::Feature::DISK);
        }

        // Register with IndexFactory
        IndexFactory::Instance().Register(full_name, create_func, feature_set);

        LOG_KNOWHERE_INFO_ << "Registered plugin " << info.name << " as " << full_name;

        registered_plugins_[name] = full_name;
        return Status::success();
    }

    // Unregister a plugin
    Status
    UnregisterPlugin(const std::string& name) {
        auto it = registered_plugins_.find(name);
        if (it == registered_plugins_.end()) {
            return Status::invalid_args_in_format("Plugin {} not registered", name);
        }

        // Note: Current IndexFactory doesn't support unregistration
        // This is a limitation of the current design
        // For now, just remove from our map
        registered_plugins_.erase(it);

        return Status::success();
    }

    // Check if a plugin is registered
    bool
    IsPluginRegistered(const std::string& name) const {
        return registered_plugins_.find(name) != registered_plugins_.end();
    }

    // Get the full registered name for a plugin
    std::string
    GetRegisteredName(const std::string& name) const {
        auto it = registered_plugins_.find(name);
        if (it != registered_plugins_.end()) {
            return it->second;
        }
        return "";
    }

    // Load and register all plugins from a directory
    Status
    LoadAndRegisterPlugins(const std::string& directory) {
        // First load all plugins
        auto status = PluginLoader::Instance().LoadPluginsFromDirectory(directory);
        if (!status.ok()) {
            return status;
        }

        // Then register them
        auto plugins = PluginLoader::Instance().ListPlugins();
        for (const auto& info : plugins) {
            auto reg_status = RegisterPlugin(info.name);
            if (!reg_status.ok()) {
                LOG_KNOWHERE_WARNING_ << "Failed to register plugin " << info.name << ": " << reg_status.what();
            }
        }

        return Status::success();
    }

    // Get information about all registered plugins
    std::vector<std::pair<std::string, plugin::PluginInfo>>
    GetRegisteredPlugins() const {
        std::vector<std::pair<std::string, plugin::PluginInfo>> result;

        for (const auto& [name, full_name] : registered_plugins_) {
            auto* factory = PluginLoader::Instance().GetPluginFactory(name);
            if (factory) {
                result.emplace_back(full_name, factory->GetPluginInfo());
            }
        }

        return result;
    }

 private:
    PluginFactory() = default;

    std::unordered_map<std::string, std::string> registered_plugins_;  // plugin name -> registered name
};

// Helper function to initialize plugins
inline Status
InitializePlugins(const std::string& plugin_directory = "") {
    if (plugin_directory.empty()) {
        // Try default locations
        std::vector<std::string> default_dirs = {"/usr/local/lib/knowhere/plugins", "/usr/lib/knowhere/plugins",
                                                 "./plugins",
                                                 std::string(std::getenv("HOME") ?: "") + "/.knowhere/plugins"};

        for (const auto& dir : default_dirs) {
            if (std::filesystem::exists(dir)) {
                LOG_KNOWHERE_INFO_ << "Loading plugins from: " << dir;
                PluginFactory::Instance().LoadAndRegisterPlugins(dir);
            }
        }
    } else {
        return PluginFactory::Instance().LoadAndRegisterPlugins(plugin_directory);
    }

    return Status::success();
}

}  // namespace knowhere::plugin
