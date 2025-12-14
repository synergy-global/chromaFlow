#include <catch2/catch_test_macros.hpp>
#include "chromaLib/ChromaFeatureExtractor.h"

using namespace ChromaFlow;

static AudioTensor makeSine(int samples, float freqHz, int sampleRate) {
    AudioTensor a;
    a.numSamples = samples;
    a.numChannels = 1;
    a.data = Eigen::VectorXf::Zero(samples);
    const float dt = 1.0f / sampleRate;
    for (int n = 0; n < samples; ++n) {
        a.data(n) = std::sin(2.0f * 3.14159265358979323846f * freqHz * n * dt) * 0.5f;
    }
    return a;
}

#ifdef CHROMAFLOW_USE_AUDIOFFT
TEST_CASE("FeatureExtractor outputs have positive entries", "[FeatureExtractor]") {
    const int sr = 48000;
    const int fft = 512;
    FeatureExtractor extractor(sr, fft, /*numMfcc*/13,
        {"mfcc","spectral_centroid","spectral_rolloff","spectral_bandwidth","spectral_energy","spectral_brightness","spectral_flatness","zcr"});

    // Use a simple sine wave to ensure non-zero spectral content
    AudioTensor input = makeSine(1024, 440.0f, sr);
    auto ft = extractor.extractFeatures(input);

    REQUIRE(ft.data.rows() == 1);
    REQUIRE(ft.data.cols() > 0);
    // Not all features are guaranteed > 0 (e.g., MFCCs can be negative),
    // but at least one scalar (energy/centroid/rolloff/brightness/flatness/ZCR) should be > 0.
    bool anyPositive = (ft.data.array() > 0.0f).any();
    REQUIRE(anyPositive);
}
#endif

TEST_CASE("DynamicsSummarizer outputs present", "[FeatureExtractor]") {
    AudioTensor input = makeSine(2048, 220.0f, 48000);

    DynamicsSummarizer rms(DynamicsSummarizer::DynamicsType::RMS);
    auto ftRms = rms.extractFeatures(input);
    REQUIRE(ftRms.data.rows() == 1);
    REQUIRE(ftRms.data.cols() == 1);
    // RMS in dB may be negative; just ensure it’s finite and not NaN
    REQUIRE(std::isfinite(ftRms.data(0,0)));

    DynamicsSummarizer peak(DynamicsSummarizer::DynamicsType::PeakLevel);
    auto ftPeak = peak.extractFeatures(input);
    REQUIRE(ftPeak.data.rows() == 1);
    REQUIRE(ftPeak.data.cols() == 1);
    REQUIRE(std::isfinite(ftPeak.data(0,0)));
}
