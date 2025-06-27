/**
 * Test program for the Knowhere plugin system
 */

#include <iostream>
#include <random>
#include <vector>

#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "knowhere/plugin/plugin_factory.h"

using namespace knowhere;

// Generate random vectors for testing
std::vector<float>
GenerateRandomVectors(int num_vectors, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    std::vector<float> data(num_vectors * dim);
    for (auto& val : data) {
        val = dis(gen);
    }
    return data;
}

int
main(int argc, char* argv[]) {
    // Initialize logging
    knowhere::KnowhereConfig::InitLog();

    std::cout << "=== Knowhere Plugin System Test ===" << std::endl;

    // 1. Load plugins from directory
    std::string plugin_dir = "./plugins";
    if (argc > 1) {
        plugin_dir = argv[1];
    }

    std::cout << "\n1. Loading plugins from: " << plugin_dir << std::endl;
    auto status = plugin::PluginFactory::Instance().LoadAndRegisterPlugins(plugin_dir);
    if (!status.ok()) {
        std::cerr << "Failed to load plugins: " << status.what() << std::endl;
        // Continue anyway - might not have plugins yet
    }

    // 2. List loaded plugins
    std::cout << "\n2. Registered plugins:" << std::endl;
    auto plugins = plugin::PluginFactory::Instance().GetRegisteredPlugins();
    if (plugins.empty()) {
        std::cout << "   No plugins loaded" << std::endl;
    } else {
        for (const auto& [name, info] : plugins) {
            std::cout << "   - " << name << " (" << info.name << " v" << info.version << " by " << info.author << ")"
                      << std::endl;
            std::cout << "     " << info.description << std::endl;
        }
    }

    // 3. Try to create and use a plugin index
    if (!plugins.empty()) {
        std::cout << "\n3. Testing plugin functionality" << std::endl;

        // Use the first available plugin
        auto plugin_name = plugins[0].first;
        std::cout << "   Using plugin: " << plugin_name << std::endl;

        // Create index through factory
        auto index_result = IndexFactory::Instance().Create(plugin_name);
        if (!index_result.has_value()) {
            std::cerr << "   Failed to create index: " << index_result.error() << std::endl;
            return 1;
        }

        auto index = std::move(index_result.value());
        std::cout << "   Created index of type: " << index->Type() << std::endl;

        // Create configuration
        auto config = index->CreateConfig();
        Json config_json = {{"dim", 128}, {"metric_type", "L2"}, {"k", 10}};
        config->Update(config_json);

        // Generate test data
        int num_train = 1000;
        int num_query = 10;
        int dim = 128;

        std::cout << "\n4. Building index with " << num_train << " vectors" << std::endl;
        auto train_data = GenerateRandomVectors(num_train, dim);

        DataSet train_dataset;
        train_dataset.SetRows(num_train);
        train_dataset.SetDim(dim);
        train_dataset.SetTensor(train_data.data());
        train_dataset.SetIsOwner(false);

        // Build index
        status = index->Build(train_dataset, *config);
        if (!status.ok()) {
            std::cerr << "   Failed to build index: " << status.what() << std::endl;
            return 1;
        }
        std::cout << "   Index built successfully" << std::endl;
        std::cout << "   Index size: " << index->Size() << " vectors" << std::endl;

        // Test search
        std::cout << "\n5. Testing search with " << num_query << " queries" << std::endl;
        auto query_data = GenerateRandomVectors(num_query, dim);

        DataSet query_dataset;
        query_dataset.SetRows(num_query);
        query_dataset.SetDim(dim);
        query_dataset.SetTensor(query_data.data());
        query_dataset.SetIsOwner(false);

        BitsetView empty_bitset;
        auto search_result = index->Search(query_dataset, *config, empty_bitset);

        if (!search_result.has_value()) {
            std::cerr << "   Search failed: " << search_result.error() << std::endl;
            return 1;
        }

        auto result = search_result.value();
        std::cout << "   Search completed successfully" << std::endl;
        std::cout << "   Result shape: " << result->GetRows() << " x " << result->GetDim() << std::endl;

        // Print first few results
        auto ids = result->GetIds();
        auto distances = result->GetDistances();
        int k = result->GetDim();

        std::cout << "\n   First query results (top-5):" << std::endl;
        for (int i = 0; i < std::min(5, k); ++i) {
            std::cout << "     ID: " << ids[i] << ", Distance: " << distances[i] << std::endl;
        }

        // Test serialization
        std::cout << "\n6. Testing serialization" << std::endl;
        BinarySet binset;
        status = index->Serialize(binset);
        if (!status.ok()) {
            std::cerr << "   Serialization failed: " << status.what() << std::endl;
            return 1;
        }
        std::cout << "   Serialization successful" << std::endl;
        std::cout << "   Binary set contains " << binset.binary_map_.size() << " entries" << std::endl;

        // Test deserialization
        std::cout << "\n7. Testing deserialization" << std::endl;
        auto new_index_result = IndexFactory::Instance().Create(plugin_name);
        if (!new_index_result.has_value()) {
            std::cerr << "   Failed to create new index for deserialization" << std::endl;
            return 1;
        }

        auto new_index = std::move(new_index_result.value());
        status = new_index->Deserialize(binset, *config);
        if (!status.ok()) {
            std::cerr << "   Deserialization failed: " << status.what() << std::endl;
            return 1;
        }
        std::cout << "   Deserialization successful" << std::endl;
        std::cout << "   Restored index size: " << new_index->Size() << " vectors" << std::endl;

        // Plugin-specific features
        if (auto plugin_index = dynamic_cast<plugin::IPluginIndex*>(index.get())) {
            std::cout << "\n8. Plugin-specific information:" << std::endl;

            // Health check
            auto health = plugin_index->HealthCheck();
            std::string health_str = "UNKNOWN";
            switch (health) {
                case plugin::PluginStatus::HEALTHY:
                    health_str = "HEALTHY";
                    break;
                case plugin::PluginStatus::DEGRADED:
                    health_str = "DEGRADED";
                    break;
                case plugin::PluginStatus::UNHEALTHY:
                    health_str = "UNHEALTHY";
                    break;
                default:
                    break;
            }
            std::cout << "   Health status: " << health_str << std::endl;

            // Features
            auto features = plugin_index->GetFeatures();
            std::cout << "   Supported features:" << std::endl;
            std::cout << "     - GPU support: " << (features.supports_gpu ? "Yes" : "No") << std::endl;
            std::cout << "     - MMap support: " << (features.supports_mmap ? "Yes" : "No") << std::endl;
            std::cout << "     - Range search: " << (features.supports_range_search ? "Yes" : "No") << std::endl;
            std::cout << "     - Iterator: " << (features.supports_iterator ? "Yes" : "No") << std::endl;
            std::cout << "     - Disk storage: " << (features.supports_disk_storage ? "Yes" : "No") << std::endl;

            std::cout << "   Supported metrics: ";
            for (const auto& metric : features.supported_metrics) {
                std::cout << metric << " ";
            }
            std::cout << std::endl;

            std::cout << "   Supported data types: ";
            for (const auto& dtype : features.supported_data_types) {
                std::cout << dtype << " ";
            }
            std::cout << std::endl;

            // Optional metrics
            auto metrics_result = plugin_index->GetMetrics();
            if (metrics_result.has_value()) {
                std::cout << "   Plugin metrics: " << metrics_result.value().dump(2) << std::endl;
            }
        }
    } else {
        std::cout << "\n3. No plugins to test. Build and install a plugin first:" << std::endl;
        std::cout << "   cd examples/simple_vector_plugin" << std::endl;
        std::cout << "   mkdir build && cd build" << std::endl;
        std::cout << "   cmake .." << std::endl;
        std::cout << "   make" << std::endl;
        std::cout << "   cp simple_vector.so ../../plugins/" << std::endl;
    }

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}
