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
#include "knowhere/comp/local_file_manager.h"
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

const std::string bpath = "/data/liuli/gist/gist_base.fbin";
const std::string qpath = "/data/liuli/gist/gist_query.fbin";
const std::string gpath = "./dist_test/gt.fbin";
float q_set[960] = {
    0.0117, 0.0115, 0.0087, 0.01,   0.0785, 0.1,    0.0784, 0.053,  0.0524, 0.0819, 0.0658, 0.058,  0.0159, 0.017,
    0.0461, 0.0242, 0.0084, 0.0064, 0.0072, 0.0102, 0.0304, 0.0679, 0.0589, 0.0571, 0.0333, 0.0786, 0.0892, 0.0423,
    0.0138, 0.0133, 0.029,  0.0219, 0.009,  0.0122, 0.0107, 0.0108, 0.0266, 0.0385, 0.0571, 0.052,  0.0355, 0.0488,
    0.0692, 0.0531, 0.0144, 0.0143, 0.0149, 0.025,  0.0171, 0.0161, 0.0106, 0.0324, 0.0271, 0.0458, 0.0531, 0.0624,
    0.0316, 0.0608, 0.0661, 0.0813, 0.0186, 0.0227, 0.0116, 0.0337, 0.0247, 0.0118, 0.0107, 0.0639, 0.0395, 0.0403,
    0.0525, 0.0958, 0.0551, 0.0676, 0.0858, 0.1749, 0.0244, 0.0281, 0.0087, 0.0512, 0.0149, 0.0086, 0.0124, 0.0356,
    0.0328, 0.0387, 0.0463, 0.0489, 0.0492, 0.0641, 0.0705, 0.1164, 0.0206, 0.0162, 0.0125, 0.0402, 0.0135, 0.0113,
    0.0074, 0.0118, 0.033,  0.0298, 0.0365, 0.042,  0.0441, 0.0519, 0.0659, 0.0527, 0.0139, 0.0162, 0.0151, 0.0224,
    0.0118, 0.0088, 0.0078, 0.0109, 0.0373, 0.0705, 0.0628, 0.0391, 0.0242, 0.0558, 0.0721, 0.0448, 0.0168, 0.0134,
    0.0331, 0.0208, 0.0041, 0.009,  0.0165, 0.0117, 0.0962, 0.1359, 0.1215, 0.088,  0.0639, 0.1174, 0.097,  0.0703,
    0.0126, 0.0219, 0.0746, 0.0422, 0.0037, 0.0084, 0.01,   0.0132, 0.032,  0.0613, 0.0751, 0.0914, 0.0429, 0.0641,
    0.0747, 0.0466, 0.0087, 0.0286, 0.0515, 0.0279, 0.0073, 0.0149, 0.0225, 0.0182, 0.0181, 0.0176, 0.0862, 0.064,
    0.0275, 0.029,  0.0558, 0.0605, 0.0114, 0.0145, 0.0207, 0.0285, 0.0208, 0.0191, 0.012,  0.045,  0.0234, 0.0231,
    0.0533, 0.0818, 0.0397, 0.0243, 0.0389, 0.1082, 0.0221, 0.0303, 0.016,  0.0534, 0.0271, 0.0189, 0.0164, 0.0881,
    0.0437, 0.017,  0.0282, 0.139,  0.0601, 0.0391, 0.0371, 0.2031, 0.0362, 0.0341, 0.0119, 0.0745, 0.0167, 0.0053,
    0.0104, 0.0456, 0.0222, 0.0201, 0.0156, 0.0458, 0.0431, 0.0402, 0.0444, 0.1346, 0.0206, 0.0192, 0.015,  0.0468,
    0.0103, 0.0066, 0.012,  0.0179, 0.0185, 0.0195, 0.044,  0.0329, 0.0298, 0.0706, 0.0955, 0.0581, 0.0097, 0.0157,
    0.0319, 0.0406, 0.0074, 0.0098, 0.0103, 0.0116, 0.0407, 0.083,  0.0675, 0.0356, 0.0352, 0.0849, 0.074,  0.0575,
    0.008,  0.0105, 0.0502, 0.0379, 0.0099, 0.0213, 0.0365, 0.0247, 0.0481, 0.0781, 0.0942, 0.084,  0.0417, 0.0525,
    0.0729, 0.0664, 0.0101, 0.0231, 0.0868, 0.0646, 0.0338, 0.0216, 0.019,  0.037,  0.0274, 0.0446, 0.0435, 0.1009,
    0.0525, 0.0677, 0.0575, 0.0464, 0.0227, 0.0417, 0.0475, 0.024,  0.0608, 0.0334, 0.0368, 0.0763, 0.0533, 0.0219,
    0.0608, 0.1125, 0.0674, 0.0218, 0.0527, 0.1312, 0.0589, 0.0331, 0.0216, 0.0653, 0.0221, 0.0174, 0.0307, 0.0303,
    0.0273, 0.0294, 0.0628, 0.0338, 0.0401, 0.0437, 0.059,  0.0761, 0.0286, 0.0189, 0.0418, 0.0696, 0.008,  0.0079,
    0.0095, 0.0111, 0.0818, 0.1053, 0.087,  0.0531, 0.0518, 0.0886, 0.0751, 0.0689, 0.0158, 0.0182, 0.0516, 0.0286,
    0.0062, 0.0048, 0.0068, 0.0097, 0.0337, 0.0694, 0.0721, 0.0618, 0.0365, 0.0824, 0.1105, 0.0488, 0.0139, 0.0127,
    0.0314, 0.0209, 0.0103, 0.0085, 0.01,   0.0086, 0.0292, 0.0371, 0.0693, 0.0521, 0.0312, 0.0465, 0.0836, 0.0486,
    0.0141, 0.0119, 0.0146, 0.0228, 0.016,  0.0123, 0.0101, 0.0303, 0.0274, 0.0497, 0.0706, 0.0674, 0.03,   0.0573,
    0.0547, 0.0882, 0.0157, 0.0201, 0.0119, 0.0353, 0.0225, 0.0112, 0.0097, 0.06,   0.0373, 0.0441, 0.0594, 0.0887,
    0.0494, 0.066,  0.0827, 0.184,  0.023,  0.0241, 0.0098, 0.0501, 0.0139, 0.0064, 0.0112, 0.0338, 0.0315, 0.0393,
    0.0474, 0.0409, 0.0426, 0.066,  0.0732, 0.125,  0.0224, 0.0155, 0.0121, 0.0373, 0.0112, 0.0063, 0.0075, 0.0112,
    0.032,  0.0283, 0.048,  0.0412, 0.0419, 0.0583, 0.0784, 0.0516, 0.0135, 0.0142, 0.0169, 0.0195, 0.0097, 0.0057,
    0.0059, 0.008,  0.0382, 0.0748, 0.0619, 0.0505, 0.0271, 0.0609, 0.0845, 0.0551, 0.0158, 0.012,  0.0369, 0.0213,
    0.0062, 0.0144, 0.0228, 0.0113, 0.104,  0.1485, 0.1449, 0.1063, 0.0659, 0.1302, 0.1167, 0.0963, 0.0152, 0.0256,
    0.0822, 0.0454, 0.0047, 0.0056, 0.011,  0.0101, 0.035,  0.0665, 0.0954, 0.1171, 0.0459, 0.0704, 0.0955, 0.0715,
    0.0097, 0.029,  0.0574, 0.0251, 0.0085, 0.0102, 0.02,   0.0148, 0.0248, 0.0185, 0.0928, 0.0701, 0.03,   0.0338,
    0.0749, 0.0749, 0.0149, 0.0174, 0.021,  0.0332, 0.0182, 0.0122, 0.0083, 0.0419, 0.0252, 0.024,  0.0531, 0.0762,
    0.0489, 0.0247, 0.033,  0.1258, 0.0175, 0.0256, 0.0131, 0.0524, 0.0261, 0.0149, 0.0157, 0.0861, 0.0434, 0.0242,
    0.0362, 0.1396, 0.0565, 0.0427, 0.0431, 0.2136, 0.0345, 0.0251, 0.0143, 0.0757, 0.0146, 0.006,  0.0093, 0.0411,
    0.0251, 0.0187, 0.031,  0.047,  0.0351, 0.053,  0.0541, 0.1344, 0.0167, 0.0176, 0.0149, 0.0439, 0.0092, 0.0055,
    0.0136, 0.0184, 0.0186, 0.0233, 0.0678, 0.0405, 0.0248, 0.0789, 0.1125, 0.0682, 0.009,  0.0144, 0.0338, 0.0389,
    0.0066, 0.0129, 0.0116, 0.0091, 0.0416, 0.0913, 0.0806, 0.0461, 0.0373, 0.0956, 0.0782, 0.0602, 0.0092, 0.0132,
    0.0552, 0.0419, 0.0095, 0.0188, 0.0403, 0.0282, 0.0528, 0.0827, 0.1135, 0.1069, 0.0432, 0.0556, 0.0898, 0.0818,
    0.0119, 0.029,  0.1051, 0.0709, 0.0328, 0.0158, 0.0191, 0.0363, 0.028,  0.0451, 0.0487, 0.1089, 0.0558, 0.0726,
    0.0727, 0.0525, 0.0207, 0.046,  0.0584, 0.0255, 0.0552, 0.0263, 0.0332, 0.0733, 0.0475, 0.0252, 0.0662, 0.112,
    0.0622, 0.0339, 0.066,  0.1363, 0.0512, 0.0276, 0.0249, 0.0683, 0.0199, 0.0159, 0.0293, 0.0286, 0.0281, 0.0352,
    0.0785, 0.0371, 0.039,  0.0509, 0.0765, 0.0821, 0.025,  0.0152, 0.0471, 0.0753, 0.0077, 0.0092, 0.0085, 0.0087,
    0.0895, 0.1157, 0.1134, 0.0741, 0.0529, 0.1015, 0.098,  0.1066, 0.0133, 0.0174, 0.0599, 0.0287, 0.0067, 0.0081,
    0.0082, 0.0077, 0.0332, 0.0759, 0.0883, 0.088,  0.0408, 0.1076, 0.1387, 0.0743, 0.0108, 0.0129, 0.0383, 0.0214,
    0.0063, 0.0075, 0.0116, 0.0074, 0.0303, 0.039,  0.083,  0.0561, 0.0311, 0.0644, 0.1124, 0.0502, 0.0117, 0.0105,
    0.0166, 0.0223, 0.0134, 0.0114, 0.0124, 0.0274, 0.0259, 0.0529, 0.0872, 0.0576, 0.0298, 0.0676, 0.0937, 0.0814,
    0.0133, 0.0178, 0.0112, 0.0341, 0.0208, 0.0121, 0.0124, 0.0528, 0.0295, 0.0446, 0.0989, 0.0666, 0.0437, 0.076,
    0.1093, 0.1461, 0.0232, 0.0193, 0.0136, 0.0484, 0.012,  0.0075, 0.0122, 0.0309, 0.03,   0.044,  0.0796, 0.0408,
    0.0343, 0.0696, 0.0906, 0.1152, 0.0183, 0.0108, 0.0177, 0.0365, 0.0102, 0.0057, 0.0079, 0.0101, 0.0322, 0.0309,
    0.0716, 0.0523, 0.0341, 0.0646, 0.1177, 0.0614, 0.0117, 0.0132, 0.019,  0.0203, 0.0098, 0.0066, 0.0067, 0.0076,
    0.0413, 0.0847, 0.0974, 0.077,  0.0297, 0.0705, 0.1156, 0.0871, 0.0121, 0.0123, 0.0413, 0.0233, 0.0061, 0.0122,
    0.0228, 0.011,  0.1219, 0.171,  0.1641, 0.1407, 0.0748, 0.1586, 0.1563, 0.1508, 0.0171, 0.0343, 0.1044, 0.0531,
    0.0052, 0.0062, 0.0121, 0.0115, 0.0393, 0.0698, 0.1048, 0.1629, 0.0452, 0.0914, 0.1193, 0.1164, 0.0072, 0.034,
    0.0732, 0.0309, 0.0054, 0.0085, 0.0198, 0.0139, 0.0307, 0.0216, 0.1018, 0.0796, 0.0345, 0.0454, 0.0958, 0.0869,
    0.0096, 0.0189, 0.0264, 0.0361, 0.0147, 0.0117, 0.0084, 0.0344, 0.0325, 0.0327, 0.055,  0.0685, 0.049,  0.0302,
    0.0409, 0.1359, 0.0159, 0.0162, 0.0128, 0.0539, 0.0154, 0.013,  0.0152, 0.0684, 0.031,  0.0289, 0.0493, 0.1177,
    0.0468, 0.056,  0.0674, 0.2025, 0.0269, 0.0158, 0.0184, 0.0677, 0.0133, 0.0097, 0.0096, 0.0332, 0.0235, 0.0247,
    0.066,  0.047,  0.0223, 0.0684, 0.0817, 0.122,  0.0107, 0.0142, 0.0186, 0.042,  0.0092, 0.0066, 0.0155, 0.0173,
    0.0241, 0.0281, 0.1046, 0.071,  0.0204, 0.0969, 0.1524, 0.0947, 0.0095, 0.0133, 0.0394, 0.0391, 0.007,  0.0129,
    0.0136, 0.0099, 0.05,   0.1057, 0.0946, 0.0848, 0.043,  0.1177, 0.0907, 0.0817, 0.0088, 0.0182, 0.0642, 0.0486,
    0.0117, 0.0227, 0.0547, 0.0372, 0.0646, 0.0933, 0.1581, 0.1529, 0.0495, 0.0595, 0.1293, 0.1124, 0.0113, 0.0368,
    0.1317, 0.0753, 0.0268, 0.013,  0.0218, 0.0346, 0.0275, 0.0453, 0.0694, 0.1228, 0.053,  0.0839, 0.1111, 0.0679,
    0.0214, 0.0564, 0.0766, 0.0238, 0.0396, 0.0164, 0.0232, 0.0558, 0.0389, 0.0357, 0.0815, 0.1093, 0.0478, 0.067,
    0.1038, 0.1442, 0.0374, 0.0213, 0.0197, 0.0601, 0.0142, 0.0173, 0.0328, 0.0207, 0.03,   0.0512, 0.1142, 0.0505,
    0.0337, 0.0719, 0.1242, 0.1062, 0.0234, 0.0124, 0.0603, 0.0866};
auto query_ds = knowhere::GenDataSet(1, 960, q_set);

knowhere::Index<knowhere::IndexNode>
CreatIndex(const std::string& index_type, int32_t version, knowhere::Json json, bool from_raw) {
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto idx = knowhere::IndexFactory::Instance().Create(index_type, version, diskann_index_pack);
    if (from_raw) {
        REQUIRE(idx.Build(knowhere::DataSet{}, json) == knowhere::Status::success);
    } else {
        knowhere::BinarySet binarySet;
        idx.Serialize(binarySet);
        idx.Deserialize(binarySet, json);
    }
    return idx;
}

TEST_CASE("Test Mem Index With Float Vector", "[float metrics]") {
    using Catch::Approx;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2);
    auto topk = GENERATE(as<int64_t>{}, 100);
    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
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

    auto disk_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = "/home/knowhere/build/disk_test";
        json["data_path"] = bpath;
        json["pq_code_budget_gb"] = sizeof(float) * 960 * 1000000 * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb"] = sizeof(float) * 960 * 1000000 * 0.125 / (1024 * 1024 * 1024);
        json["search_list_size"] = 100;
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
            make_tuple(knowhere::IndexEnum::INDEX_DISKANN, disk_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen2),
            // make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        // const std::string gpath = "/data/liuli/glove/gt";
        // const std::string gpath = "/home/knowhere/build/glove_gt.fbin";
        std::cout << 1 << std::endl;
        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        auto idx = CreatIndex(name, version, json, false);
        std::cout << 2 << std::endl;
        // auto query_ds = GenDataSetFromFile(qpath);
        auto results = idx.RangeSearch(*query_ds, json, nullptr).value();
        std::cout << "searched size: " << results->GetRows() << std::endl;
        std::cout << 3 << std::endl;
        // auto gt = GenGTFromFile(gpath);
        auto train_ds = GenDataSetFromFile(bpath);
        auto result_range = knowhere::BruteForce::RangeSearch(train_ds, query_ds, json, nullptr).value();
        std::cout << "GT size: " << result_range->GetRows() << std::endl;
        auto ap = GetRangeSearchRecall(*result_range, *results);

        std::cout << "test recall:  " << ap << std::endl;
    }
}
