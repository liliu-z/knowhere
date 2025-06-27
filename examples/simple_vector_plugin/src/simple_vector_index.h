/**
 * Example plugin: SimpleVector - A brute-force vector search implementation
 */

#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "knowhere/plugin/plugin_interface.h"
#include "knowhere/utils/distances.h"

namespace simple_vector_plugin {

using namespace knowhere;

// Configuration for SimpleVector index
class SimpleVectorConfig : public plugin::IPluginConfig {
 public:
    CFG_INT dim;
    CFG_STRING metric_type;
    CFG_INT k;

    KNOWHERE_CONFIG_DECLARE_FIELD(dim).description("vector dimension").set_default(128).set_range(1, 32768).for_train();

    KNOWHERE_CONFIG_DECLARE_FIELD(metric_type).description("metric type").set_default("L2").for_train().for_search();

    KNOWHERE_CONFIG_DECLARE_FIELD(k).description("topk").set_default(10).set_range(1, 1024).for_search();

    Status
    Validate() const override {
        if (dim.value() <= 0) {
            return Status::invalid_args_in_format("dim must be positive, got {}", dim.value());
        }
        if (metric_type.value() != "L2" && metric_type.value() != "IP") {
            return Status::invalid_args_in_format("unsupported metric type: {}", metric_type.value());
        }
        return Status::success();
    }

    Json
    GetDefaultConfig() const override {
        return Json{{"dim", 128}, {"metric_type", "L2"}, {"k", 10}};
    }
};

// Simple brute-force vector index
class SimpleVectorIndex : public plugin::IPluginIndex {
 public:
    SimpleVectorIndex() = default;

    // Plugin metadata
    plugin::PluginInfo
    GetPluginInfo() const override {
        return plugin::PluginInfo{.name = "SimpleVector",
                                  .version = "1.0.0",
                                  .author = "Knowhere Example",
                                  .description = "A simple brute-force vector search plugin",
                                  .license = "MIT",
                                  .api_version = KNOWHERE_PLUGIN_API_VERSION};
    }

    plugin::PluginStatus
    HealthCheck() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (vectors_.empty()) {
            return plugin::PluginStatus::HEALTHY;  // Empty but healthy
        }
        // Could add more checks here
        return plugin::PluginStatus::HEALTHY;
    }

    plugin::PluginFeatures
    GetFeatures() const override {
        return plugin::PluginFeatures{.supports_gpu = false,
                                      .supports_mmap = false,
                                      .supports_range_search = false,
                                      .supports_iterator = false,
                                      .supports_disk_storage = false,
                                      .supported_metrics = {"L2", "IP"},
                                      .supported_data_types = {"float32"}};
    }

    // Build index (for SimpleVector, just store the vectors)
    Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto config = static_cast<const SimpleVectorConfig&>(cfg);

        if (!HasRawData(dataset)) {
            return Status::invalid_args("dataset must have raw data");
        }

        auto dim = dataset.GetDim();
        auto rows = dataset.GetRows();
        auto data = dataset.GetTensor();

        if (dim != config.dim.value()) {
            return Status::invalid_args_in_format("dimension mismatch: expected {}, got {}", config.dim.value(), dim);
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Store configuration
        dim_ = dim;
        metric_type_ = config.metric_type.value();

        // Copy vectors
        vectors_.resize(rows * dim);
        std::memcpy(vectors_.data(), data, rows * dim * sizeof(float));
        num_vectors_ = rows;

        LOG_KNOWHERE_INFO_ << "Built SimpleVector index with " << rows << " vectors, dim=" << dim;

        return Status::success();
    }

    // Search implementation
    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto config = static_cast<const SimpleVectorConfig&>(cfg);

        if (!HasRawData(dataset)) {
            return expected<DataSetPtr>::Err(Status::invalid_args("dataset must have raw data"), nullptr);
        }

        auto dim = dataset.GetDim();
        auto nq = dataset.GetRows();
        auto queries = static_cast<const float*>(dataset.GetTensor());
        auto k = config.k.value();

        if (dim != dim_) {
            return expected<DataSetPtr>::Err(
                Status::invalid_args_in_format("dimension mismatch: expected {}, got {}", dim_, dim), nullptr);
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if (num_vectors_ == 0) {
            // Return empty result
            auto result = std::make_shared<DataSet>();
            result->SetRows(nq);
            result->SetDim(0);
            return result;
        }

        // Allocate result buffers
        auto ids = std::make_unique<int64_t[]>(nq * k);
        auto distances = std::make_unique<float[]>(nq * k);

        // Brute-force search for each query
        for (int64_t q = 0; q < nq; ++q) {
            const float* query = queries + q * dim_;

            // Compute distances to all vectors
            std::vector<std::pair<float, int64_t>> dist_idx;
            dist_idx.reserve(num_vectors_);

            for (int64_t i = 0; i < num_vectors_; ++i) {
                // Skip if filtered by bitset
                if (bitset.test(i)) {
                    continue;
                }

                float dist = 0.0f;
                const float* vec = vectors_.data() + i * dim_;

                if (metric_type_ == "L2") {
                    // L2 distance
                    for (int64_t d = 0; d < dim_; ++d) {
                        float diff = query[d] - vec[d];
                        dist += diff * diff;
                    }
                } else {  // IP
                    // Inner product (negative for max-heap behavior)
                    for (int64_t d = 0; d < dim_; ++d) {
                        dist -= query[d] * vec[d];
                    }
                }

                dist_idx.emplace_back(dist, i);
            }

            // Find top-k
            int64_t valid_k = std::min(k, static_cast<int64_t>(dist_idx.size()));
            std::partial_sort(dist_idx.begin(), dist_idx.begin() + valid_k, dist_idx.end());

            // Fill results
            for (int64_t i = 0; i < valid_k; ++i) {
                ids[q * k + i] = dist_idx[i].second;
                distances[q * k + i] = dist_idx[i].first;
            }

            // Fill remaining with -1
            for (int64_t i = valid_k; i < k; ++i) {
                ids[q * k + i] = -1;
                distances[q * k + i] = std::numeric_limits<float>::max();
            }
        }

        // Create result dataset
        auto result = std::make_shared<DataSet>();
        result->SetRows(nq);
        result->SetDim(k);
        result->SetIds(ids.get());
        result->SetDistances(distances.get());
        result->SetIsOwner(true);
        ids.release();
        distances.release();

        return result;
    }

    // Other required methods
    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented("RangeSearch not implemented"), nullptr);
    }

    expected<std::shared_ptr<AnnIterator>>
    AnnIterator(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return expected<std::shared_ptr<AnnIterator>>::Err(Status::not_implemented("AnnIterator not implemented"),
                                                           nullptr);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        auto ids = dataset.GetIds();
        auto rows = dataset.GetRows();

        std::lock_guard<std::mutex> lock(mutex_);

        auto result_data = std::make_unique<float[]>(rows * dim_);

        for (int64_t i = 0; i < rows; ++i) {
            auto id = ids[i];
            if (id < 0 || id >= num_vectors_) {
                return expected<DataSetPtr>::Err(
                    Status::invalid_args_in_format("id {} out of range [0, {})", id, num_vectors_), nullptr);
            }
            std::memcpy(result_data.get() + i * dim_, vectors_.data() + id * dim_, dim_ * sizeof(float));
        }

        auto result = std::make_shared<DataSet>();
        result->SetRows(rows);
        result->SetDim(dim_);
        result->SetTensor(result_data.get());
        result->SetIsOwner(true);
        result_data.release();

        return result;
    }

    bool
    HasRawData(knowhere::MetricType metric_type) const override {
        return true;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        Json meta = {{"num_vectors", num_vectors_},
                     {"dim", dim_},
                     {"metric_type", metric_type_},
                     {"index_type", "SimpleVector"},
                     {"memory_usage", num_vectors_ * dim_ * sizeof(float)}};

        auto result = std::make_shared<DataSet>();
        result->SetMeta(meta);
        return result;
    }

    Status
    Serialize(BinarySet& binset) const override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Serialize metadata
        Json meta = {{"num_vectors", num_vectors_}, {"dim", dim_}, {"metric_type", metric_type_}};

        auto meta_str = meta.dump();
        binset.Append("meta", meta_str.data(), meta_str.size());

        // Serialize vectors
        if (num_vectors_ > 0) {
            binset.Append("vectors", vectors_.data(), num_vectors_ * dim_ * sizeof(float));
        }

        return Status::success();
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Deserialize metadata
        auto meta_binary = binset.GetByName("meta");
        if (!meta_binary) {
            return Status::invalid_args("missing meta in binary set");
        }

        std::string meta_str(static_cast<const char*>(meta_binary->data.get()), meta_binary->size);
        auto meta = Json::parse(meta_str);

        num_vectors_ = meta["num_vectors"];
        dim_ = meta["dim"];
        metric_type_ = meta["metric_type"];

        // Deserialize vectors
        if (num_vectors_ > 0) {
            auto vectors_binary = binset.GetByName("vectors");
            if (!vectors_binary) {
                return Status::invalid_args("missing vectors in binary set");
            }

            vectors_.resize(num_vectors_ * dim_);
            std::memcpy(vectors_.data(), vectors_binary->data.get(), num_vectors_ * dim_ * sizeof(float));
        }

        return Status::success();
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        return Status::not_implemented("DeserializeFromFile not implemented");
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<SimpleVectorConfig>();
    }

    int64_t
    Dim() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return dim_;
    }

    int64_t
    Size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_vectors_;
    }

    int64_t
    Count() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_vectors_;
    }

    std::string
    Type() const override {
        return "SimpleVector";
    }

 private:
    mutable std::mutex mutex_;
    std::vector<float> vectors_;
    int64_t num_vectors_ = 0;
    int64_t dim_ = 0;
    std::string metric_type_ = "L2";
};

}  // namespace simple_vector_plugin
