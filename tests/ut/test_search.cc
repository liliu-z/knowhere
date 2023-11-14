// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <fstream>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/utils/binary_distances.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.99f;
}  // namespace

float*
ReadFromFile(const std::string& path, int32_t& rows, int32_t& dim) {
    std::ifstream in(path, std::ios::binary);
    in.read((char*)&rows, sizeof(int32_t));
    in.read((char*)&dim, sizeof(int32_t));
    std::cout << "rows: " << rows << ", dim: " << dim << std::endl;
    float* data = new float[rows * dim];
    in.read((char*)data, rows * dim * sizeof(float));
    in.close();
    return data;
}

knowhere::DataSetPtr
GenDataSetFromFile(const std::string& path) {
    // read from file
    int32_t rows, dim;
    float* data = ReadFromFile(path, rows, dim);
    auto ds = knowhere::GenDataSet(rows, dim, data);
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
GenGTFromFile(const std::string& path) {
    int32_t rows, dim;
    int32_t* data = (int32_t*)ReadFromFile(path, rows, dim);
    int64_t* ids = new int64_t[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ids[i] = data[i];
    }
    delete[] data;
    float* dis = new float[1];
    auto ds = knowhere::GenResultDataSet(rows, dim, ids, dis);
    ds->SetIsOwner(true);
    return ds;
}

TEST_CASE("Test Mem Index With Float Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::IP);
    auto topk = GENERATE(as<int64_t>{}, 100);
    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 1024;
        json[knowhere::indexparam::NPROBE] = 60;
        return json;
    };

    auto ivfflatcc_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    auto ivfsq_gen = ivfflat_gen;

    auto flat_gen = base_gen;

    auto ivfpq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 4;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto scann_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::NPROBE] = 14;
        json[knowhere::indexparam::REORDER_K] = 500;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        return json;
    };

    auto scann_gen2 = [scann_gen]() {
        knowhere::Json json = scann_gen();
        json[knowhere::indexparam::WITH_RAW_DATA] = false;
        return json;
    };

    auto hnsw_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 200;
        return json;
    };

    // const auto train_ds = GenDataSet(nb, dim);
    // const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    // auto gt = knowhere::BruteForce::Search(train_ds, query_ds, conf, nullptr);

    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen2),
            // make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        const std::string bpath = "/data/liuli/glove/glove_base.fbin";
        const std::string qpath = "/data/liuli/glove/glove_query.fbin";
        // const std::string gpath = "/data/liuli/glove/gt";
        const std::string gpath = "/home/knowhere/build/glove_gt.fbin";

        auto idx = knowhere::IndexFactory::Instance().Create(name, version);
        auto cfg_json = gen().dump();

        knowhere::Json json = knowhere::Json::parse(cfg_json);

        auto train_ds = GenDataSetFromFile(bpath);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

        auto query_ds = GenDataSetFromFile(qpath);
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());

        auto gt = GenGTFromFile(gpath);
        float recall = GetKNNRecall(*gt, *results.value());

        std::cout << "test recall:  " << recall << std::endl;

        // serdes tet
        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

        results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());

        recall = GetKNNRecall(*gt, *results.value());
        std::cout << "test recall:  " << recall << std::endl;
    }
}

