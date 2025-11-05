#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "chromaLib/ChromaBaseClasses.h"

using namespace ChromaFlow;
using Catch::Approx;

TEST_CASE("ChromaUtils activation functions", "[ChromaUtils][ActivationFunctions]") {
    SECTION("Sigmoid function") {
        REQUIRE(ChromaUtils::sigmoid(0.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::sigmoid(1.0f) == Approx(0.7311f).epsilon(0.001f));
        REQUIRE(ChromaUtils::sigmoid(-1.0f) == Approx(0.2689f).epsilon(0.001f));
        
        // Test extreme values
        REQUIRE(ChromaUtils::sigmoid(10.0f) == Approx(1.0f).epsilon(0.001f));
        // Near zero, use absolute margin since relative epsilon isn't meaningful
        REQUIRE(ChromaUtils::sigmoid(-10.0f) == Approx(0.0f).margin(1e-4f));
    }
    
    SECTION("Tanh function") {
        REQUIRE(ChromaUtils::tanh(0.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::tanh(1.0f) == Approx(0.7616f).epsilon(0.001f));
        REQUIRE(ChromaUtils::tanh(-1.0f) == Approx(-0.7616f).epsilon(0.001f));
    }
    
    SECTION("ReLU function") {
        REQUIRE(ChromaUtils::relu(0.0f) == 0.0f);
        REQUIRE(ChromaUtils::relu(1.0f) == 1.0f);
        REQUIRE(ChromaUtils::relu(-1.0f) == 0.0f);
        REQUIRE(ChromaUtils::relu(5.5f) == 5.5f);
    }
}

TEST_CASE("ChromaUtils utility functions", "[ChromaUtils][Utilities]") {
    SECTION("Clip function") {
        REQUIRE(ChromaUtils::clip(0.5f, 0.0f, 1.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::clip(-0.5f, 0.0f, 1.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::clip(1.5f, 0.0f, 1.0f) == Approx(1.0f));
    }
    
    SECTION("Clip vector function") {
        std::vector<float> input = {-1.0f, 0.5f, 1.5f, 0.0f, 2.0f};
        std::vector<float> result = ChromaUtils::clipVector(input, 0.0f, 1.0f);
        
        REQUIRE(result.size() == 5);
        REQUIRE(result[0] == Approx(0.0f));
        REQUIRE(result[1] == Approx(0.5f));
        REQUIRE(result[2] == Approx(1.0f));
        REQUIRE(result[3] == Approx(0.0f));
        REQUIRE(result[4] == Approx(1.0f));
    }
    
    SECTION("Unit normalization") {
        // 0..100
        REQUIRE(ChromaUtils::mapToUnit(0.0f, 0.0f, 100.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::mapToUnit(50.0f, 0.0f, 100.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::mapToUnit(100.0f, 0.0f, 100.0f) == Approx(1.0f));

        // 0..30
        REQUIRE(ChromaUtils::mapToUnit(0.0f, 0.0f, 30.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::mapToUnit(15.0f, 0.0f, 30.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::mapToUnit(30.0f, 0.0f, 30.0f) == Approx(1.0f));

        // 1..5
        REQUIRE(ChromaUtils::mapToUnit(1.0f, 1.0f, 5.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::mapToUnit(3.0f, 1.0f, 5.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::mapToUnit(5.0f, 1.0f, 5.0f) == Approx(1.0f));

        // Reversed range 5..1
        REQUIRE(ChromaUtils::mapToUnit(1.0f, 5.0f, 1.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::mapToUnit(3.0f, 5.0f, 1.0f) == Approx(0.5f));
        REQUIRE(ChromaUtils::mapToUnit(5.0f, 5.0f, 1.0f) == Approx(1.0f));

        // Clamping beyond bounds
        REQUIRE(ChromaUtils::mapToUnit(-10.0f, 0.0f, 100.0f) == Approx(0.0f));
        REQUIRE(ChromaUtils::mapToUnit(110.0f, 0.0f, 100.0f) == Approx(1.0f));

        // Degenerate range
        REQUIRE(ChromaUtils::mapToUnit(42.0f, 10.0f, 10.0f) == Approx(0.0f));
    }
}

TEST_CASE("ChromaUtils audio statistics", "[ChromaUtils][AudioStats]") {
    std::vector<float> audioBlock = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f};
    
    SECTION("RMS calculation") {
        float rms = ChromaUtils::calculateRMS(audioBlock);
        // RMS = sqrt((1 + 1 + 0.25 + 0.25 + 0) / 5) = sqrt(2.5/5) = sqrt(0.5)
        REQUIRE(rms == Approx(0.7071f).epsilon(0.001f));
    }
    
    SECTION("Mean absolute calculation") {
        float meanAbs = ChromaUtils::calculateMeanAbs(audioBlock);
        // Mean abs = (1 + 1 + 0.5 + 0.5 + 0) / 5 = 3/5 = 0.6
        REQUIRE(meanAbs == Approx(0.6f));
    }
    
    SECTION("Standard deviation calculation") {
        float std = ChromaUtils::calculateStd(audioBlock);
        // This is a more complex calculation, just verify it's reasonable
        REQUIRE(std > 0.0f);
        REQUIRE(std < 1.0f);
    }
    
    SECTION("Skewness calculation") {
        std::vector<float> symmetricData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float skewness = ChromaUtils::calculateSkewness(symmetricData);
        // Symmetric data should have skewness close to 0
        REQUIRE(skewness == Approx(0.0f).epsilon(0.1f));
    }
    
    SECTION("Kurtosis calculation") {
        std::vector<float> normalData = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        float kurtosis = ChromaUtils::calculateKurtosis(normalData);
        // This function returns excess kurtosis (kurtosis - 3). For this platykurtic sample,
        // the expected excess kurtosis is approximately -1.3.
        REQUIRE(kurtosis == Approx(-1.3f).margin(0.05f));
    }
}

TEST_CASE("ChromaUtils edge cases", "[ChromaUtils][EdgeCases]") {
    SECTION("Empty vector handling") {
        std::vector<float> empty;
        
        // These should handle empty vectors gracefully
        REQUIRE(ChromaUtils::calculateRMS(empty) == 0.0f);
        REQUIRE(ChromaUtils::calculateMeanAbs(empty) == 0.0f);
    }
    
    SECTION("Single element vector") {
        std::vector<float> single = {0.5f};
        
        REQUIRE(ChromaUtils::calculateRMS(single) == Approx(0.5f));
        REQUIRE(ChromaUtils::calculateMeanAbs(single) == Approx(0.5f));
        REQUIRE(ChromaUtils::calculateStd(single) == 0.0f);
    }
}