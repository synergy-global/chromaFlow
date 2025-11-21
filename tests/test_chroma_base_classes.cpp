#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "chromaLib/ChromaBaseClasses.h"
#include "chromaLib/ChromaLayers.h"

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

TEST_CASE("NNAnalyzer parameter counting and memory estimates", "[NNAnalyzer]") {
    SECTION("Dense layer with bias") {
        const size_t inputSize = 128;
        const size_t outputSize = 64;
        const bool hasBias = true;
        const size_t expected = 128 * 64 + 64; // 8256
        REQUIRE(NNAnalyzer::countTrainableParameters(inputSize, outputSize, hasBias) == expected);
    }

    SECTION("Conv-like without bias") {
        const size_t inputSize = 32;
        const size_t outputSize = 16;
        const bool hasBias = false;
        const size_t expected = 32 * 16; // 512
        REQUIRE(NNAnalyzer::countTrainableParameters(inputSize, outputSize, hasBias) == expected);
    }

    SECTION("SGD memory footprint (float)") {
        const size_t params = 1000;
        const bool isAdam = false;
        const size_t bytes = NNAnalyzer::calculateTotalMemory(params, isAdam, 4);
        // params + 1 state vector (momentum)
        REQUIRE(bytes == (1000 * 4 + 1000 * 4)); // 8000
    }

    SECTION("Adam memory footprint (float)") {
        const size_t params = 1000;
        const bool isAdam = true;
        const size_t bytes = NNAnalyzer::calculateTotalMemory(params, isAdam, 4);
        // params + 2 state vectors (m, v)
        REQUIRE(bytes == (1000 * 4 + 2000 * 4)); // 12000
    }

    SECTION("Adam memory footprint (double)") {
        const size_t params = 500;
        const bool isAdam = true;
        const size_t bytes = NNAnalyzer::calculateTotalMemory(params, isAdam, 8);
        REQUIRE(bytes == (500 * 8 + 1000 * 8)); // 12000
    }
}

TEST_CASE("NNAnalyzer model-level counting aggregates across layers", "[NNAnalyzer]") {
    using namespace ChromaFlow;
    using std::shared_ptr;
    using std::make_shared;

    // Build a simple model: Conv(4x3) + Dense(5->3) + RNN(5->3)
    std::vector<std::shared_ptr<DifferentiableModule>> modules;
    modules.emplace_back(make_shared<convolutionalLayer>(/*inputChannels*/1, /*outputChannels*/4, /*kernelSize*/3));
    modules.emplace_back(make_shared<denseLayer>(/*inputSize*/5, /*outputSize*/3, ActivationType::LeakyRelu, /*useLayerNorm*/true));
    modules.emplace_back(make_shared<RNNCell>(/*inputSize*/5, /*hiddenSize*/3));

    // Expected counts reflect current layer implementations:
    // Conv: kernel is not trainable by default (uses implicit ones), only bias vector is trainable
    //       => biases_.size() = 4
    // Dense: weights (3x5) + gamma (3) + beta (3) + alpha (5) = 15 + 3 + 3 + 5 = 26
    // RNN:   W_x (3x5) + W_h (3x3) + b (3) = 15 + 9 + 3 = 27
    const size_t expected_total = 4 + 26 + 27; // 57

    const size_t counted = NNAnalyzer::countTrainableParameters(modules);
    REQUIRE(counted == expected_total);

    // Memory calculation for SGD (two copies per param), float32 (4 bytes)
    const size_t mem_sgd = NNAnalyzer::calculateTotalMemoryForModel(modules, /*isAdam*/false, /*dataTypeSizeBytes*/4);
    REQUIRE(mem_sgd == expected_total * 1 * 4 + expected_total * 1 * 4);

    // Memory calculation for Adam (four copies per param), float32 (4 bytes)
    const size_t mem_adam = NNAnalyzer::calculateTotalMemoryForModel(modules, /*isAdam*/true, /*dataTypeSizeBytes*/4);
    REQUIRE(mem_adam == expected_total * 1 * 4 + expected_total * 2 * 4);
}

TEST_CASE("NNAnalyzer parameter counting and memory calculation", "[NNAnalyzer]") {
    SECTION("countTrainableParameters for dense layer with bias") {
        size_t inputSize = 128;
        size_t outputSize = 64;
        bool hasBias = true;
        
        size_t result = NNAnalyzer::countTrainableParameters(inputSize, outputSize, hasBias);
        size_t expected = (128 * 64) + 64; // 8256 parameters
        
        REQUIRE(result == expected);
    }
    
    SECTION("countTrainableParameters for conv layer without bias") {
        size_t inputSize = 32;
        size_t outputSize = 16;
        bool hasBias = false;
        
        size_t result = NNAnalyzer::countTrainableParameters(inputSize, outputSize, hasBias);
        size_t expected = 32 * 16; // 512 parameters
        
        REQUIRE(result == expected);
    }
    
    SECTION("calculateTotalMemory for SGD optimizer") {
        size_t totalParameters = 1000;
        bool isAdam = false;
        size_t sampleTypeSize = 4;
        
        size_t result = NNAnalyzer::calculateTotalMemory(totalParameters, isAdam, sampleTypeSize);
        size_t expected = (1000 * 4) + (1000 * 4); // 8000 bytes (params + momentum)
        
        REQUIRE(result == expected);
    }
    
    SECTION("calculateTotalMemory for Adam optimizer") {
        size_t totalParameters = 1000;
        bool isAdam = true;
        size_t sampleTypeSize = 4;
        
        size_t result = NNAnalyzer::calculateTotalMemory(totalParameters, isAdam, sampleTypeSize);
        size_t expected = (1000 * 4) + (2000 * 4); // 12000 bytes (params + momentum + variance)
        
        REQUIRE(result == expected);
    }
    
    SECTION("calculateTotalMemory with double precision") {
        size_t totalParameters = 500;
        bool isAdam = true;
        size_t sampleTypeSize = 8;
        
        size_t result = NNAnalyzer::calculateTotalMemory(totalParameters, isAdam, sampleTypeSize);
        size_t expected = (500 * 8) + (1000 * 8); // 12000 bytes (params + momentum + variance with double)
        
        REQUIRE(result == expected);
    }
}