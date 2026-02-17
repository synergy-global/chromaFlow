﻿#include <catch2/catch_test_macros.hpp>
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

    class TestRMSHomeostasisLoss : public RMSHomeostasisLoss
    {
    public:
        using RMSHomeostasisLoss::RMSHomeostasisLoss;
        using RMSHomeostasisLoss::calculate;

        FeatureTensor calculate(const FeatureTensor &input,
                                const FeatureTensor &output) const override
        {
            (void)input;
            return calculate(output);
        }

        void setTarget(float t) override
        {
            target = t;
        }
    };

    class TestSpectralTiltLoss : public SpectralTiltLoss
    {
    public:
        using SpectralTiltLoss::SpectralTiltLoss;
        using SpectralTiltLoss::calculate;

        FeatureTensor calculate(const FeatureTensor &input,
                                const FeatureTensor &output) const override
        {
            (void)input;
            return calculate(output);
        }

        void setTarget(float t) override
        {
            target = t;
        }
    };

    class TestSpectralFluxLoss : public SpectralFluxLoss
    {
    public:
        using SpectralFluxLoss::SpectralFluxLoss;
        using SpectralFluxLoss::calculate;

        FeatureTensor calculate(const FeatureTensor &input,
                                const FeatureTensor &output) const override
        {
            (void)input;
            return calculate(output);
        }

        void setTarget(float t) override
        {
            target = t;
        }
    };

    class TestHarmonicEnergyLoss : public HarmonicEnergyLoss
    {
    public:
        using HarmonicEnergyLoss::HarmonicEnergyLoss;
        using HarmonicEnergyLoss::calculate;

        FeatureTensor calculate(const FeatureTensor &input,
                                const FeatureTensor &output) const override
        {
            (void)input;
            return calculate(output);
        }

        void setTarget(float t) override
        {
            target = t;
        }
    };
} // namespace

TEST_CASE("RMSHomeostasisLoss gradient vanishes at target RMS", "[Losses]")
{
    TestRMSHomeostasisLoss loss;
    loss.setTarget(1.0f);

    FeatureTensor y = makeRow({1.0f, 1.0f, 1.0f, 1.0f});
    FeatureTensor g = loss.calculate(y);

    REQUIRE(g.data.rows() == 1);
    REQUIRE(g.data.cols() == y.data.cols());

    for (int c = 0; c < g.data.cols(); ++c)
        REQUIRE(g.data(0, c) == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("SpectralTiltLoss gradient direction matches target tilt", "[Losses]")
{
    TestSpectralTiltLoss loss;

    FeatureTensor out = makeRow({1.0f, 2.0f});
    Eigen::VectorXf lowHigh(2);
    lowHigh << 1.0f, 2.0f;
    float tilt = std::log(lowHigh(1)) - std::log(lowHigh(0));

    loss.setTarget(tilt);
    FeatureTensor g_eq = loss.calculate(out);

    loss.setTarget(tilt + 0.5f);
    FeatureTensor g_highTilt = loss.calculate(out);

    REQUIRE(g_eq.data(0, 0) == Approx(0.0f).margin(1e-4f));
    REQUIRE(g_eq.data(0, 1) == Approx(0.0f).margin(1e-4f));

    REQUIRE(g_highTilt.data(0, 0) < 0.0f);
    REQUIRE(g_highTilt.data(0, 1) > 0.0f);
}

TEST_CASE("SpectralFluxLoss returns zero on first frame and diff on second", "[Losses]")
{
    TestSpectralFluxLoss loss;

    FeatureTensor f1 = makeRow({0.1f, 0.2f, 0.3f});
    FeatureTensor g1 = loss.calculate(f1);

    REQUIRE(g1.data.rows() == 1);
    REQUIRE(g1.data.cols() == 3);
    for (int c = 0; c < 3; ++c)
        REQUIRE(g1.data(0, c) == Approx(0.0f).margin(1e-6f));

    FeatureTensor f2 = makeRow({0.2f, 0.1f, 0.4f});
    FeatureTensor g2 = loss.calculate(f2);

    REQUIRE(g2.data.rows() == 1);
    REQUIRE(g2.data.cols() == 3);
    REQUIRE(g2.data(0, 0) == Catch::Approx(0.1f).margin(1e-6f));
    REQUIRE(g2.data(0, 1) == Catch::Approx(-0.1f).margin(1e-6f));
    REQUIRE(g2.data(0, 2) == Catch::Approx(0.1f).margin(1e-6f));
}

TEST_CASE("EnergyConservationLoss pushes toward matching input/output energy", "[Losses]")
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

TEST_CASE("HarmonicEnergyLoss zero gradient below maxEnergy and positive above", "[Losses]")
{
    TestHarmonicEnergyLoss loss;
    loss.setTargetEnergy(0.5f);

    FeatureTensor low = makeRow({0.1f, 0.2f});
    FeatureTensor g_low = loss.calculate(low);

    for (int c = 0; c < g_low.data.cols(); ++c)
        REQUIRE(g_low.data(0, c) == Catch::Approx(0.0f).margin(1e-6f));

    FeatureTensor high = makeRow({1.0f, 1.0f});
    FeatureTensor g_high = loss.calculate(high);

    REQUIRE(g_high.data(0, 0) > 0.0f);
    REQUIRE(g_high.data(0, 1) > 0.0f);
}
