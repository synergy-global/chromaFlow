#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include "chromaLib/ChromaLayers.h"

using namespace ChromaFlow::Layers;

TEST_CASE("Layer forwards produce positive outputs", "[Layers]") {
    // Prepare a simple FeatureTensor input with small positive values
    ChromaFlow::FeatureTensor in;
    in.numSamples = 1;
    in.features = 8;
    in.data = Eigen::MatrixXf::Zero(1, 8);
    in.data.setConstant(0.1f);

    SECTION("convolutionalLayer forward > 0") {
        convolutionalLayer conv( /*kernelSize*/3);
        Eigen::VectorXf k(3); k << 0.2f, 0.2f, 0.2f;
        conv.setKernel(k);
        conv.setBias(0.05f);

        // Build a 1D column FeatureTensor input expected by conv (N x 1)
        ChromaFlow::FeatureTensor convIn;
        convIn.numSamples = 16;
        convIn.features = 1;
        convIn.data = Eigen::MatrixXf::Ones(16, 1) * 0.1f;

        auto out = conv.forward(convIn);
        REQUIRE(out.data.size() > 0);
        // Check at least one element is > 0
        bool anyPositive = (out.data.array() > 0.0f).any();
        REQUIRE(anyPositive);
        {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
            std::cout << "convolutionalLayer last output: "
                      << out.data.col(0).format(fmt)
                      << std::endl;
        }
    }

    SECTION("denseLayer forward > 0") {
        DenseLayer dense(/*inputSize*/8, /*outputSize*/5);
        auto out = dense.forward(in);
        REQUIRE(out.data.rows() == 1);
        REQUIRE(out.data.cols() == 5);
        bool anyPositive = (out.data.array() > 0.0f).any();
        REQUIRE(anyPositive);
        {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
            std::cout << "denseLayer last output: "
                      << out.data.row(0).transpose().format(fmt)
                      << std::endl;
        }
    }

    SECTION("attentionLayer forward > 0 (single-vector path)") {
        attentionLayer att(/*inputSize*/8, /*numHeads*/4, /*outputSize*/8);
        auto out = att.forward(in);
        REQUIRE(out.data.rows() == 1);
        REQUIRE(out.data.cols() == 8);
        
        bool anyPositive = (out.data.array() > 0.0f).any();
        REQUIRE(anyPositive);
        {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
            std::cout << "attentionLayer last output: "
                      << out.data.row(0).transpose().format(fmt)
                      << std::endl;
        }
    }

    SECTION("RNNCell forward > 0") {
        RNNCell rnn(/*inputSize*/8, /*hiddenSize*/6);
        auto out = rnn.forward(in);
        REQUIRE(out.data.rows() == 1);
        REQUIRE(out.data.cols() == 6);
        bool anyPositive = (out.data.array() > 0.0f).any();
        REQUIRE(anyPositive);
        // print last output for debugging
        {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
            std::cout << "RNNCell last output: "
                      << out.data.row(0).transpose().format(fmt)
                      << std::endl;
        }
    }
}
