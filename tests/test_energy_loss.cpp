#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "chromaLib/ChromaLossFunctions.h"

using namespace ChromaFlow;
using namespace ChromaFlow::LossFunctions;
using Catch::Approx;

namespace
{
    FeatureTensor makeRow(const std::initializer_list<float> &vals)
    {
        FeatureTensor t;
        t.numSamples = 1;
        t.features = static_cast<int>(vals.size());
        t.data.resize(1, t.features);
        int c = 0;
        for (float v : vals)
            t.data(0, c++) = v;
        return t;
    }
}

TEST_CASE("EnergyConservationLoss zero gradient when energies match", "[Losses]")
{
    EnergyConservationLoss loss;

    FeatureTensor x = makeRow({1.0f, 2.0f});
    FeatureTensor y = makeRow({1.0f, 2.0f});

    FeatureTensor g = loss.calculate(y, x);

    REQUIRE(g.data.rows() == 1);
    REQUIRE(g.data.cols() == 2);
    REQUIRE(g.data(0, 0) == Approx(0.0f).margin(1e-6f));
    REQUIRE(g.data(0, 1) == Approx(0.0f).margin(1e-6f));
}

TEST_CASE("EnergyConservationLoss positive gradient when output has higher energy", "[Losses]")
{
    EnergyConservationLoss loss;

    FeatureTensor x = makeRow({1.0f, 1.0f});
    FeatureTensor y = makeRow({2.0f, 2.0f});

    FeatureTensor g = loss.calculate(y, x);

    REQUIRE(g.data.rows() == 1);
    REQUIRE(g.data.cols() == 2);
    REQUIRE(g.data(0, 0) > 0.0f);
    REQUIRE(g.data(0, 1) > 0.0f);
}

