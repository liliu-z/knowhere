/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dlfcn.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "knowhere/log.h"
#include "knowhere/plugin/plugin_interface.h"

namespace knowhere::plugin {

// Loaded plugin information
struct LoadedPlugin {
    std::string path;
    void* handle;
    std::unique_ptr<IPluginFactory> factory;
    std::unique_ptr<IPluginLifecycle> lifecycle;
    PluginInfo info;
};

class PluginLoader {
 public:
    static PluginLoader&
    Instance() {
        static PluginLoader instance;
        return instance;
    }

    // Load all plugins from a directory
    Status
    LoadPluginsFromDirectory(const std::string& directory) {
        if (!std::filesystem::exists(directory)) {
            LOG_KNOWHERE_WARNING_ << "Plugin directory does not exist: " << directory;
            return Status::invalid_args;
        }

        LOG_KNOWHERE_INFO_ << "Loading plugins from: " << directory;

        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                if (ext == ".so" || ext == ".dylib" || ext == ".dll") {
                    auto status = LoadPlugin(entry.path().string());
                    if (!status.ok()) {
                        LOG_KNOWHERE_WARNING_ << "Failed to load plugin: " << entry.path()
                                              << ", error: " << status.what();
                    }
                }
            }
        }

        LOG_KNOWHERE_INFO_ << "Loaded " << loaded_plugins_.size() << " plugins";
        return Status::success();
    }

    // Load a single plugin
    Status
    LoadPlugin(const std::string& path) {
        LOG_KNOWHERE_INFO_ << "Loading plugin: " << path;

        // Check if already loaded
        if (loaded_plugins_.find(path) != loaded_plugins_.end()) {
            LOG_KNOWHERE_WARNING_ << "Plugin already loaded: " << path;
            return Status::invalid_args;
        }

        // Load the shared library
        void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            return Status::invalid_args_in_format("Failed to load plugin: {}", dlerror());
        }

        try {
            // Get API version
            auto get_version = (uint32_t (*)())dlsym(handle, "GetKnowherePluginAPIVersion");
            if (!get_version) {
                dlclose(handle);
                return Status::invalid_args_in_format("Plugin missing GetKnowherePluginAPIVersion: {}", dlerror());
            }

            uint32_t plugin_api_version = get_version();
            if (plugin_api_version != KNOWHERE_PLUGIN_API_VERSION) {
                dlclose(handle);
                return Status::invalid_args_in_format("Plugin API version mismatch: expected {}, got {}",
                                                      KNOWHERE_PLUGIN_API_VERSION, plugin_api_version);
            }

            // Create factory
            auto create_factory = (IPluginFactory * (*)()) dlsym(handle, "CreateKnowherePluginFactory");
            auto destroy_factory = (void (*)(IPluginFactory*))dlsym(handle, "DestroyKnowherePluginFactory");

            if (!create_factory || !destroy_factory) {
                dlclose(handle);
                return Status::invalid_args_in_format("Plugin missing factory functions: {}", dlerror());
            }

            // Create factory instance
            auto factory = std::unique_ptr<IPluginFactory>(create_factory());
            if (!factory) {
                dlclose(handle);
                return Status::invalid_args("Failed to create plugin factory");
            }

            // Get plugin info
            PluginInfo info = factory->GetPluginInfo();
            LOG_KNOWHERE_INFO_ << "Loaded plugin: " << info.name << " v" << info.version << " by " << info.author;

            // Optional: Get lifecycle handler
            std::unique_ptr<IPluginLifecycle> lifecycle;
            auto get_lifecycle = (IPluginLifecycle * (*)()) dlsym(handle, "GetKnowherePluginLifecycle");
            if (get_lifecycle) {
                lifecycle.reset(get_lifecycle());
                if (lifecycle) {
                    auto status = lifecycle->OnLoad();
                    if (!status.ok()) {
                        LOG_KNOWHERE_WARNING_ << "Plugin OnLoad failed: " << status.what();
                    }
                }
            }

            // Store loaded plugin
            LoadedPlugin plugin{.path = path,
                                .handle = handle,
                                .factory = std::move(factory),
                                .lifecycle = std::move(lifecycle),
                                .info = info};

            loaded_plugins_[path] = std::move(plugin);
            plugin_names_[info.name] = path;

            return Status::success();

        } catch (const std::exception& e) {
            dlclose(handle);
            return Status::invalid_args_in_format("Plugin loading exception: {}", e.what());
        }
    }

    // Unload a plugin
    Status
    UnloadPlugin(const std::string& name) {
        auto name_it = plugin_names_.find(name);
        if (name_it == plugin_names_.end()) {
            return Status::invalid_args_in_format("Plugin not found: {}", name);
        }

        auto path = name_it->second;
        auto it = loaded_plugins_.find(path);
        if (it == loaded_plugins_.end()) {
            return Status::invalid_args;
        }

        // Call lifecycle OnUnload if available
        if (it->second.lifecycle) {
            it->second.lifecycle->OnUnload();
        }

        // Close the library
        dlclose(it->second.handle);

        // Remove from maps
        plugin_names_.erase(name);
        loaded_plugins_.erase(it);

        LOG_KNOWHERE_INFO_ << "Unloaded plugin: " << name;
        return Status::success();
    }

    // Get plugin factory by name
    IPluginFactory*
    GetPluginFactory(const std::string& name) {
        auto name_it = plugin_names_.find(name);
        if (name_it == plugin_names_.end()) {
            return nullptr;
        }

        auto it = loaded_plugins_.find(name_it->second);
        if (it == loaded_plugins_.end()) {
            return nullptr;
        }

        return it->second.factory.get();
    }

    // List all loaded plugins
    std::vector<PluginInfo>
    ListPlugins() const {
        std::vector<PluginInfo> result;
        for (const auto& [path, plugin] : loaded_plugins_) {
            result.push_back(plugin.info);
        }
        return result;
    }

    // Unload all plugins
    void
    UnloadAll() {
        for (auto& [path, plugin] : loaded_plugins_) {
            if (plugin.lifecycle) {
                plugin.lifecycle->OnUnload();
            }
            dlclose(plugin.handle);
        }
        loaded_plugins_.clear();
        plugin_names_.clear();
    }

    ~PluginLoader() {
        UnloadAll();
    }

 private:
    PluginLoader() = default;
    PluginLoader(const PluginLoader&) = delete;
    PluginLoader&
    operator=(const PluginLoader&) = delete;

    std::unordered_map<std::string, LoadedPlugin> loaded_plugins_;  // path -> plugin
    std::unordered_map<std::string, std::string> plugin_names_;     // name -> path
};
