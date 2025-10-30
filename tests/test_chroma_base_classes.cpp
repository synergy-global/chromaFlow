#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "chromaLib/ChromaBaseClasses.h"

using namespace ChromaFlow;
using Catch::Approx;

TEST_CASE("AudioTensor basic functionality", "[AudioTensor]") {
    AudioTensor audio;
    audio.numSamples = 1024;
    audio.numChannels = 2;
    audio.data = Eigen::VectorXf::Zero(1024 * 2);
    
    SECTION("AudioTensor initialization") {
        REQUIRE(audio.numSamples == 1024);
        REQUIRE(audio.numChannels == 2);
        REQUIRE(audio.data.size() == 1024 * 2);
    }
    
    SECTION("AudioTensor data manipulation") {
        audio.data.setConstant(0.5f);
        REQUIRE(audio.data(0) == Approx(0.5f));
        REQUIRE(audio.data(1023) == Approx(0.5f));
    }
}

TEST_CASE("FeatureTensor basic functionality", "[FeatureTensor]") {
    FeatureTensor features;
    features.numSamples = 512;
    features.features = 13; // MFCC features
    features.data = Eigen::MatrixXf::Zero(512, 13);
    
    SECTION("FeatureTensor initialization") {
        REQUIRE(features.numSamples == 512);
        REQUIRE(features.features == 13);
        REQUIRE(features.data.rows() == 512);
        REQUIRE(features.data.cols() == 13);
    }
    
    SECTION("FeatureTensor data access") {
        features.data(0, 0) = 1.0f;
        features.data(511, 12) = 2.0f;
        
        REQUIRE(features.data(0, 0) == Approx(1.0f));
        REQUIRE(features.data(511, 12) == Approx(2.0f));
    }
}

TEST_CASE("ParamTensor basic functionality", "[ParamTensor]") {
    ParamTensor params;
    
    SECTION("ParamTensor parameter storage") {
        params.data["learning_rate"] = 0.001f;
        params.data["momentum"] = 0.9f;
        params.data["weight_decay"] = 0.0001f;
        
        REQUIRE(params.data["learning_rate"] == Approx(0.001f));
        REQUIRE(params.data["momentum"] == Approx(0.9f));
        REQUIRE(params.data["weight_decay"] == Approx(0.0001f));
        REQUIRE(params.data.size() == 3);
    }
}

TEST_CASE("WeightTensor basic functionality", "[WeightTensor]") {
    WeightTensor weights;
    weights.data = Eigen::MatrixXf::Random(10, 5);
    
    SECTION("WeightTensor dimensions") {
        REQUIRE(weights.data.rows() == 10);
        REQUIRE(weights.data.cols() == 5);
    }
    
    SECTION("WeightTensor operations") {
        // Test matrix operations
        Eigen::MatrixXf input = Eigen::MatrixXf::Ones(5, 1);
        Eigen::MatrixXf output = weights.data * input;
        
        REQUIRE(output.rows() == 10);
        REQUIRE(output.cols() == 1);
    }
}