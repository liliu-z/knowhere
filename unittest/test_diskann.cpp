// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <cmath>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <chrono>
#include <fstream>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "knowhere/feder/DiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/LocalFileManager.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 56;
constexpr float kMax = 100;
constexpr uint32_t kK = 10;
constexpr uint32_t kBigK = kNumRows * 2;
constexpr float kL2Radius = 300000;
constexpr float kL2RangeFilter = 0;
constexpr float kIPRadius = 50000;
constexpr float kIPRangeFilter = std::numeric_limits<float>::max();
constexpr float kDisLossTolerance = 0.5;

constexpr uint32_t kLargeDimNumRows = 1000;
constexpr uint32_t kLargeDimNumQueries = 10;
constexpr uint32_t kLargeDim = 5600;
constexpr uint32_t kLargeDimBigK = kLargeDimNumRows * 2;
constexpr float kLargeDimL2Radius = 36000000;
constexpr float kLargeDimL2RangeFilter = 0;
constexpr float kLargeDimIPRadius = 400000;
constexpr float kLargeDimIPRangeFilter = std::numeric_limits<float>::max();

std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kLargeDimRawDataPath = kDir + "/large_dim_raw_data";
std::string kIpIndexDir = kDir + "/ip_index";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kLargeDimIpIndexDir = kDir + "/large_dim_ip_index";
std::string kLargeDimL2IndexDir = kDir + "/large_dim_l2_index";

const knowhere::DiskANNBuildConfig build_conf{kRawDataPath, 50, 90, 0.2, 0.2, 4, 0};
const knowhere::DiskANNBuildConfig large_dim_build_conf{kLargeDimRawDataPath, 50, 90, 0.2, 0.2, 4, 0};
const knowhere::DiskANNPrepareConfig prep_conf{4, 0.0015, false, false};
const knowhere::DiskANNQueryConfig query_conf{kK, kK * 10, 3};
const knowhere::DiskANNQueryByRangeConfig l2_range_search_conf{kL2Radius, kL2RangeFilter, true, 10, 10000, 3};
const knowhere::DiskANNQueryByRangeConfig ip_range_search_conf{kIPRadius, kIPRangeFilter, true, 10, 10000, 3};
const knowhere::DiskANNQueryByRangeConfig large_dim_l2_range_search_conf{
    kLargeDimL2Radius, kLargeDimL2RangeFilter, true, 10, 1000, 3};
const knowhere::DiskANNQueryByRangeConfig large_dim_ip_range_search_conf{
    kLargeDimIPRadius, kLargeDimIPRangeFilter, true, 10, 1000, 3};

std::random_device rd;
size_t x = rd();
std::mt19937 generator((unsigned)x);
std::uniform_real_distribution<float> distribution(-1, 1);

float*
GenData(size_t num) {
    float* data_p = new float[num];

    for (int i = 0; i < num; ++i) {
        float rnd_val = distribution(generator) * static_cast<float>(kMax);
        data_p[i] = rnd_val;
    }

    return data_p;
}

float*
GenLargeData(size_t num) {
    float max = std::numeric_limits<float>::max();
    auto data_p = GenData(num);
    for (auto i = 0; i < num; i++) {
        data_p[i] = distribution(generator) * max;
        if (!std::isnormal(data_p[i])) {
            data_p[i] = 1.0;
        }
    }
    return data_p;
}

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * num * dim);
    writer.close();
}

template <typename DiskANNConfig>
void
CheckConfigError(DiskANNConfig& config_to_test) {
    knowhere::Config cfg;
    DiskANNConfig::Set(cfg, config_to_test);
    EXPECT_THROW(DiskANNConfig::Get(cfg), knowhere::KnowhereException);
}

void
CheckDistanceError(const float* data_p, const float* query_p, const knowhere::DatasetPtr result,
                   const std::string metric, const uint32_t num_query, const uint32_t dim_query, const uint32_t topk,
                   const uint32_t row_nums, const bool is_large_dim) {
    if (is_large_dim)
        return;
    auto res_ids_p = knowhere::GetDatasetIDs(result);
    auto res_dis_p = knowhere::GetDatasetDistance(result);
    uint32_t valid_res_num = topk < row_nums ? topk : row_nums;
    for (auto q = 0; q < num_query; q++) {
        for (auto k = 0; k < valid_res_num; k++) {
            auto id_q_k = res_ids_p[q * topk + k];
            EXPECT_NE(id_q_k, -1);

            float true_dis = 0;
            if (metric == knowhere::metric::IP) {
                for (int d = 0; d < dim_query; d++) {
                    true_dis += (data_p[dim_query * id_q_k + d] * query_p[dim_query * q + d]);
                }
            } else if (metric == knowhere::metric::L2) {
                for (int d = 0; d < dim_query; d++) {
                    true_dis += ((data_p[dim_query * id_q_k + d] - query_p[dim_query * q + d]) *
                                 (data_p[dim_query * id_q_k + d] - query_p[dim_query * q + d]));
                }
            }
            EXPECT_NEAR(true_dis, res_dis_p[q * topk + k], kDisLossTolerance);
        }
    }
}

}  // namespace

class DiskANNTest : public TestWithParam<std::tuple<std::string, bool>> {
 public:
    DiskANNTest() {
        std::tie(metric_, is_large_dim_) = GetParam();
        if (!is_large_dim_) {
            dim_ = kDim;
            num_rows_ = kNumRows;
            num_queries_ = kNumQueries;
            big_k_ = kBigK;
            raw_data_ = global_raw_data_;
            query_data_ = global_query_data_;
            ground_truth_ = metric_ == knowhere::metric::L2 ? l2_ground_truth_ : ip_ground_truth_;
            range_search_ground_truth_ =
                metric_ == knowhere::metric::L2 ? l2_range_search_ground_truth_ : ip_range_search_ground_truth_;
            range_search_conf_ = metric_ == knowhere::metric::L2 ? l2_range_search_conf : ip_range_search_conf;
            radius_ = metric_ == knowhere::metric::L2 ? kL2Radius : kIPRadius;
            range_filter_ = metric_ == knowhere::metric::L2 ? kL2RangeFilter : kIPRangeFilter;
        } else {
            dim_ = kLargeDim;
            num_rows_ = kLargeDimNumRows;
            num_queries_ = kLargeDimNumQueries;
            big_k_ = kLargeDimBigK;
            raw_data_ = global_large_dim_raw_data_;
            query_data_ = global_large_dim_query_data_;
            ground_truth_ = metric_ == knowhere::metric::L2 ? large_dim_l2_ground_truth_ : large_dim_ip_ground_truth_;
            range_search_ground_truth_ = metric_ == knowhere::metric::L2 ? large_dim_l2_range_search_ground_truth_
                                                                         : large_dim_ip_range_search_ground_truth_;
            range_search_conf_ =
                metric_ == knowhere::metric::L2 ? large_dim_l2_range_search_conf : large_dim_ip_range_search_conf;
            radius_ = metric_ == knowhere::metric::L2 ? kLargeDimL2Radius : kLargeDimIPRadius;
            range_filter_ = metric_ == knowhere::metric::L2 ? kLargeDimL2RangeFilter : kLargeDimIPRangeFilter;
        }
        // InitDiskANN();
    }

    ~DiskANNTest() {
    }

    static void
    SetUpTestCase() {
        LOG_KNOWHERE_INFO_ << "Setting up the test environment for DiskANN Unittest.";
        // fs::remove_all(kDir);
        // fs::remove(kDir);

        global_raw_data_ = GenData(kNumRows * kDim);
        // global_large_dim_raw_data_ = GenData(kLargeDimNumRows * kLargeDim);
        global_query_data_ = GenData(kNumQueries * kDim);
        // global_large_dim_query_data_ = GenData(kLargeDimNumQueries * kLargeDim);

        // ip_ground_truth_ =
        //     GenGroundTruth(global_raw_data_, global_query_data_, knowhere::metric::IP, kNumRows, kDim, kNumQueries,
        //     kK);
        // l2_ground_truth_ =
        //     GenGroundTruth(global_raw_data_, global_query_data_, knowhere::metric::L2, kNumRows, kDim, kNumQueries,
        //     kK);
        // ip_range_search_ground_truth_ =
        //     GenRangeSearchGrounTruth(global_raw_data_, global_query_data_, knowhere::metric::IP, kNumRows, kDim,
        //                              kNumQueries, kIPRadius, kIPRangeFilter);
        // l2_range_search_ground_truth_ =
        //     GenRangeSearchGrounTruth(global_raw_data_, global_query_data_, knowhere::metric::L2, kNumRows, kDim,
        //  kNumQueries, kL2Radius, kL2RangeFilter);

        // large_dim_ip_ground_truth_ =
        //     GenGroundTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::IP,
        //                    kLargeDimNumRows, kLargeDim, kLargeDimNumQueries, kK);
        // large_dim_l2_ground_truth_ =
        //     GenGroundTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::L2,
        //                    kLargeDimNumRows, kLargeDim, kLargeDimNumQueries, kK);
        // large_dim_ip_range_search_ground_truth_ = GenRangeSearchGrounTruth(
        //     global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::IP, kLargeDimNumRows,
        //     kLargeDim, kLargeDimNumQueries, kLargeDimIPRadius, kLargeDimIPRangeFilter);
        // large_dim_l2_range_search_ground_truth_ = GenRangeSearchGrounTruth(
        //     global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::L2, kLargeDimNumRows,
        //     kLargeDim, kLargeDimNumQueries, kLargeDimL2Radius, kLargeDimL2RangeFilter);

        // prepare the dir
        // ASSERT_TRUE(fs::create_directory(kDir));
        // ASSERT_TRUE(fs::create_directory(kIpIndexDir));
        // ASSERT_TRUE(fs::create_directory(kL2IndexDir));
        // ASSERT_TRUE(fs::create_directory(kLargeDimL2IndexDir));
        // ASSERT_TRUE(fs::create_directory(kLargeDimIpIndexDir));

        // WriteRawDataToDisk(kRawDataPath, global_raw_data_, kNumRows, kDim);
        // WriteRawDataToDisk(kLargeDimRawDataPath, global_large_dim_raw_data_, kLargeDimNumRows, kLargeDim);

        // knowhere::Config cfg;
        // knowhere::DiskANNBuildConfig::Set(cfg, build_conf);
        // auto diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
        //     kIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        // diskann_ip->BuildAll(nullptr, cfg);
        // auto diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
        //     kL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        // diskann_l2->BuildAll(nullptr, cfg);

        // knowhere::Config large_dim_cfg;
        // knowhere::DiskANNBuildConfig::Set(large_dim_cfg, large_dim_build_conf);
        // auto large_dim_diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
        //     kLargeDimIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        // large_dim_diskann_ip->BuildAll(nullptr, large_dim_cfg);
        // auto large_dim_diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
        //     kLargeDimL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        // large_dim_diskann_l2->BuildAll(nullptr, large_dim_cfg);
    }

    static void
    TearDownTestCase() {
        LOG_KNOWHERE_INFO_ << "Cleaning up the test environment for DiskANN Unittest.";
        delete[] global_raw_data_;
        // delete[] global_large_dim_raw_data_;
        delete[] global_query_data_;
        // delete[] global_large_dim_query_data_;
        // Clean up the dir

        // fs::remove_all(kDir);
        // fs::remove(kDir);
    }

 protected:
    void
    InitDiskANN() {
        std::string index_dir = "";
        if (metric_ == knowhere::metric::L2) {
            index_dir = is_large_dim_ ? kLargeDimL2IndexDir : kL2IndexDir;
        } else {
            index_dir = is_large_dim_ ? kLargeDimIpIndexDir : kIpIndexDir;
        }
        diskann = std::make_unique<knowhere::IndexDiskANN<float>>(index_dir + "/diskann", metric_,
                                                                  std::make_unique<knowhere::LocalFileManager>());
    }
    static float* global_raw_data_;
    static float* global_large_dim_raw_data_;
    static float* global_query_data_;
    static float* global_large_dim_query_data_;
    static GroundTruthPtr ip_ground_truth_;
    static GroundTruthPtr l2_ground_truth_;
    static GroundTruthPtr l2_range_search_ground_truth_;
    static GroundTruthPtr ip_range_search_ground_truth_;
    static GroundTruthPtr large_dim_ip_ground_truth_;
    static GroundTruthPtr large_dim_l2_ground_truth_;
    static GroundTruthPtr large_dim_l2_range_search_ground_truth_;
    static GroundTruthPtr large_dim_ip_range_search_ground_truth_;
    std::string metric_;
    bool is_large_dim_;
    uint32_t dim_;
    uint32_t num_rows_;
    uint32_t num_queries_;
    uint32_t big_k_;
    GroundTruthPtr ground_truth_;
    GroundTruthPtr range_search_ground_truth_;
    float* raw_data_;
    float* query_data_;
    float radius_;
    float range_filter_;
    knowhere::DiskANNQueryByRangeConfig range_search_conf_;
    std::unique_ptr<knowhere::VecIndex> diskann;
};

float* DiskANNTest::global_raw_data_ = nullptr;
float* DiskANNTest::global_large_dim_raw_data_ = nullptr;
float* DiskANNTest::global_query_data_ = nullptr;
float* DiskANNTest::global_large_dim_query_data_ = nullptr;
GroundTruthPtr DiskANNTest::ip_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::ip_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_ip_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_l2_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_l2_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_ip_range_search_ground_truth_ = nullptr;

INSTANTIATE_TEST_CASE_P(DiskANNParameters, DiskANNTest,
                        Values(std::make_tuple(knowhere::metric::L2, false /* low-dimension */)));

TEST_P(DiskANNTest, knn_search_test) {
    std::string index_dir = "";
    if (metric_ == knowhere::metric::L2) {
        index_dir = is_large_dim_ ? kLargeDimL2IndexDir : kL2IndexDir;
    } else {
        index_dir = is_large_dim_ ? kLargeDimIpIndexDir : kIpIndexDir;
    }
    auto diskann1 = std::make_unique<knowhere::IndexDiskANN<float>>(index_dir + "/diskann", metric_,
                                                                    std::make_unique<knowhere::LocalFileManager>());

    knowhere::Config cfg;
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    // // test query before preparation
    // EXPECT_THROW(diskann->Query(data_set_ptr, cfg, nullptr), knowhere::KnowhereException);

    // test preparation
    // cfg.clear();
    knowhere::Config pcfg;
    knowhere::DiskANNPrepareConfig::Set(pcfg, prep_conf);
    EXPECT_TRUE(diskann1->Prepare(pcfg));

    std::vector<std::thread> threads;
    for (int i = 0; i < 40; i++) {
        std::thread t([&]() {
            auto diskann2 = std::make_unique<knowhere::IndexDiskANN<float>>(
                index_dir + "/diskann", metric_, std::make_unique<knowhere::LocalFileManager>());
            EXPECT_TRUE(diskann2->Prepare(pcfg));
            for (int i = 0; i < 10000; i++) {
                if (i % 10 == 0)
                    std::cout << "thread " << std::this_thread::get_id() << " " << i << std::endl;
                auto result = diskann2->Query(data_set_ptr, cfg, nullptr);
            }
        });

        threads.emplace_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }

    // test query
    // cfg.clear();
    // sleep for 5 s

    // auto ids = knowhere::GetDatasetIDs(result);
    // auto diss = knowhere::GetDatasetDistance(result);

    // auto recall = CheckTopKRecall(ground_truth_, ids, kK, num_queries_);
}
