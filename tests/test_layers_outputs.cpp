#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include "chromaLib/ChromaLayers.h"

using namespace ChromaFlow;

TEST_CASE("Layer forwards produce positive outputs", "[Layers]") {
    // Prepare a simple FeatureTensor input with small positive values
    FeatureTensor in;
    in.numSamples = 1;
    in.features = 8;
    in.data = Eigen::MatrixXf::Zero(1, 8);
    in.data.setConstant(0.1f);

    SECTION("convolutionalLayer forward > 0") {
        convolutionalLayer conv(/*inputChannels*/1, /*outputChannels*/4, /*kernelSize*/3);
        // Use a positive kernel and bias to encourage positive outputs
        Eigen::VectorXf k(3); k << 0.2f, 0.2f, 0.2f;
        conv.setKernel(k);
        Eigen::VectorXf b(1); b << 0.05f;
        conv.setBiases(b);

        // Build a 1D column FeatureTensor input expected by conv (N x 1)
        FeatureTensor convIn;
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
        denseLayer dense(/*inputSize*/8, /*outputSize*/5, ActivationType::LeakyRelu, /*useLayerNorm*/true);
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
        attentionLayer att(/*inputSize*/8, /*outputSize*/8, /*heads*/4, ActivationType::Linear, /*useLayerNorm*/true);
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
