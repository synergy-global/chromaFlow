#pragma once

#include "ChromaBaseClasses.h"
#include "ChromaFFT.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>
#include <cassert>

namespace ChromaFlow
{

// ================================================================
// Window
// ================================================================

enum class WindowType
{
    Hann,
    Hamming,
    Blackman
};

class Window
{
public:
    void init(size_t size, WindowType type = WindowType::Hann)
    {
        assert(size > 0);
        _size = size;
        _data.resize(size);

        constexpr float pi = 3.14159265358979323846f;

        for (size_t n = 0; n < size; ++n)
        {
            float a = 2.0f * pi * n / (size - 1);

            switch (type)
            {
                case WindowType::Hann:
                    _data[n] = 0.5f - 0.5f * std::cos(a);
                    break;

                case WindowType::Hamming:
                    _data[n] = 0.54f - 0.46f * std::cos(a);
                    break;

                case WindowType::Blackman:
                    _data[n] = 0.42f - 0.5f * std::cos(a)
                               + 0.08f * std::cos(2.0f * a);
                    break;
            }
        }

        computeNormalization();
    }

    inline void apply(float* data) const noexcept
    {
        for (size_t i = 0; i < _size; ++i)
            data[i] *= _data[i];
    }

    inline float normalization() const noexcept { return _norm; }

private:
    size_t _size = 0;
    std::vector<float> _data;
    float _norm = 1.0f;

    void computeNormalization()
    {
        float sumSq = 0.0f;
        for (float v : _data)
            sumSq += v * v;

        float rms = std::sqrt(sumSq / float(_size));
        _norm = (rms > 0.0f) ? 1.0f / rms : 1.0f;
    }
};

// ================================================================
// FeatureExtractor
// ================================================================

class FeatureExtractor : public featureExtraction
{
public:

    FeatureExtractor(int sampleRate,
                     int nFft = 512,
                     int numMfcc = 13,
                     const std::vector<std::string>& featureNames =
                     {
                         "mfcc",
                         "spectral_centroid",
                         "spectral_rolloff",
                         "spectral_bandwidth",
                         "spectral_energy",
                         "spectral_brightness",
                         "spectral_flatness",
                         "zcr"
                     },
                     WindowType winType = WindowType::Hann)
        : sample_rate(sampleRate),
          n_fft(nearestPow2(std::max(8, nFft))),
          num_mfcc(std::max(1, numMfcc)),
          requested(featureNames.begin(), featureNames.end())
    {
        fft.init(n_fft);
        complex_bins = int(fft.ComplexSize(n_fft));

        window.init(n_fft, winType);

        re_buffer.resize(complex_bins);
        im_buffer.resize(complex_bins);

        power_spectrum.resize(complex_bins);
        mag_spectrum.resize(complex_bins);
        freqs.resize(complex_bins);
        cumsum.resize(complex_bins);

        for (int k = 0; k < complex_bins; ++k)
            freqs[k] = float(k) * float(sample_rate) / float(n_fft);

        ensureMel();
        ensureDct();
    }

    FeatureTensor extractFeatures(const AudioTensor& input) override
    {
        const int inSize = int(input.data.size());

        // ---------- ZCR (raw input, no padding) ----------
        float zcr = 0.0f;
        if (requested.count("zcr") && inSize > 1)
        {
            int crossings = 0;
            float prev = input.data(0);
            for (int i = 1; i < inSize; ++i)
            {
                float curr = input.data(i);
                if ((prev >= 0 && curr < 0) ||
                    (prev < 0 && curr >= 0))
                    ++crossings;
                prev = curr;
            }
            zcr = float(crossings) / float(inSize - 1);
        }

        // ---------- FFT block ----------
        Eigen::VectorXf block = Eigen::VectorXf::Zero(n_fft);
        block.head(std::min(inSize, n_fft)) = input.data.head(std::min(inSize, n_fft));

        window.apply(block.data());
        block *= window.normalization();

        fft.fft(block.data(), re_buffer.data(), im_buffer.data());

        for (int k = 0; k < complex_bins; ++k)
        {
            float rr = re_buffer[k];
            float ii = im_buffer[k];
            float p = rr * rr + ii * ii;
            power_spectrum[k] = p;
            mag_spectrum[k] = std::sqrt(std::max(0.0f, p));
        }

        float spec_sum = mag_spectrum.sum() + 1e-6f;
        float power_sum = power_spectrum.sum() + 1e-6f;

        // ---------- Build FeatureTensor directly ----------
        FeatureTensor out;
        int maxFeatures = num_mfcc + 7;
        out.data.resize(1, maxFeatures);

        int col = 0;

        // ---------- MFCC ----------
        if (requested.count("mfcc"))
        {
            Eigen::VectorXf mel = mel_filter * power_spectrum;
            Eigen::VectorXf logmel = (mel.array() + 1e-6f).log();
            Eigen::VectorXf mfcc = dct * logmel;

            applyLifter(mfcc, 22);

            for (int i = 0; i < num_mfcc; ++i)
                out.data(0, col++) = mfcc[i];
        }

        // ---------- Spectral features ----------
        if (requested.count("spectral_centroid"))
        {
            float centroid = (freqs.array() * mag_spectrum.array()).sum() / spec_sum;
            out.data(0, col++) = centroid;
        }

        if (requested.count("spectral_rolloff"))
        {
            float acc = 0.0f;
            for (int k = 0; k < complex_bins; ++k)
            {
                acc += mag_spectrum[k];
                cumsum[k] = acc;
            }
            float target = 0.85f * cumsum[complex_bins - 1];
            int idx = 0;
            while (idx < complex_bins && cumsum[idx] < target)
                ++idx;
            out.data(0, col++) = freqs[idx];
        }

        if (requested.count("spectral_energy"))
            out.data(0, col++) = power_sum / float(n_fft);

        if (requested.count("spectral_flatness"))
        {
            auto p = power_spectrum.array() + 1e-12f;
            float geo = std::exp(p.log().mean());
            float arith = p.mean();
            out.data(0, col++) = geo / (arith + 1e-12f);
        }

        if (requested.count("zcr"))
            out.data(0, col++) = zcr;

        out.data.conservativeResize(1, col);
        out.numSamples = 1;
        out.features = col;

        return out;
    }

private:

    int sample_rate;
    int n_fft;
    int num_mfcc;
    int complex_bins;

    std::unordered_set<std::string> requested;

#ifdef __APPLE__
    ChromaFFT::AppleAccelerateFFT fft;
#else
    ChromaFFT::Radix2FFTImpl fft;
#endif

    Window window;

    Eigen::MatrixXf mel_filter;
    Eigen::MatrixXf dct;

    std::vector<float> re_buffer;
    std::vector<float> im_buffer;

    Eigen::VectorXf power_spectrum;
    Eigen::VectorXf mag_spectrum;
    Eigen::VectorXf freqs;
    Eigen::VectorXf cumsum;

    static int nearestPow2(int n)
    {
        int p = 1;
        while ((p << 1) <= n) p <<= 1;
        return p;
    }

    void applyLifter(Eigen::VectorXf& mfcc, int L)
    {
        if (L <= 0) return;
        for (int k = 0; k < mfcc.size(); ++k)
        {
            float lift = 1.0f + (L / 2.0f) *
                std::sin(float(M_PI) * k / float(L));
            mfcc[k] *= lift;
        }
    }

        static int nearestPowerOfTwo(int n)
        {
            int p = 1;
            while ((p << 1) <= n)
                p <<= 1;
            return p;
        }

        void ensureMelFilterbank()
        {
            if (mel_filterbank.rows() == mel_n_mels && mel_filterbank.cols() == complex_bins)
                return;

            const float low_freq_mel = 0.0f;
            const float high_freq_mel =
                2595.0f * std::log10(1.0f + (static_cast<float>(sample_rate) / 2.0f) / 700.0f);

            Eigen::VectorXf mel_points =
                Eigen::VectorXf::LinSpaced(mel_n_mels + 2, low_freq_mel, high_freq_mel);

            Eigen::VectorXf hz_points(mel_points.size());
            for (int i = 0; i < mel_points.size(); ++i)
            {
                hz_points[i] =
                    700.0f * (std::pow(10.0f, mel_points[i] / 2595.0f) - 1.0f);
            }

            std::vector<int> bin_points(static_cast<size_t>(mel_points.size()));
            for (int i = 0; i < hz_points.size(); ++i)
            {
                const float binF = (static_cast<float>(n_fft) + 1.0f) * hz_points[i] / static_cast<float>(sample_rate);
                int bin = static_cast<int>(std::floor(binF));
                bin = std::max(0, std::min(bin, complex_bins - 1));
                bin_points[static_cast<size_t>(i)] = bin;
            }

            mel_filterbank.resize(mel_n_mels, complex_bins);
            mel_filterbank.setZero();
            for (int m = 1; m <= mel_n_mels; ++m)
            {
                const int f_m_minus = bin_points[static_cast<size_t>(m - 1)];
                const int f_m = bin_points[static_cast<size_t>(m)];
                const int f_m_plus = bin_points[static_cast<size_t>(m + 1)];
                if (f_m_minus >= f_m_plus)
                    continue;

                for (int k = f_m_minus; k < f_m; ++k)
                {
                    float denom = std::max(1, f_m - f_m_minus);
                    mel_filterbank(m - 1, k) =
                        (static_cast<float>(k - f_m_minus) / static_cast<float>(denom));
                }

                for (int k = f_m; k < f_m_plus; ++k)
                {
                    float denom = std::max(1, f_m_plus - f_m);
                    mel_filterbank(m - 1, k) =
                        (static_cast<float>(f_m_plus - k) / static_cast<float>(denom));
                }
            }
        }

        void ensureDctMatrix()
        {
            if (dct_matrix.rows() == num_mfcc && dct_matrix.cols() == mel_n_mels)
                return;

            dct_matrix = Eigen::MatrixXf(num_mfcc, mel_n_mels);
            const float N = static_cast<float>(mel_n_mels);
            const float scale0 = std::sqrt(1.0f / (4.0f * N));
            const float scale = std::sqrt(1.0f / (2.0f * N));
            const float pi = 3.14159265358979323846f;

            for (int k = 0; k < num_mfcc; ++k)
            {
                for (int n = 0; n < mel_n_mels; ++n)
                {
                    const float angle = pi * (static_cast<float>(n) + 0.5f) * static_cast<float>(k) / N;
                    float s = std::cos(angle);
                    dct_matrix(k, n) = (k == 0 ? (2.0f * scale0) : (2.0f * scale)) * s;
                }
            }
        }
};
 
    // =======================================================================
    // DynamicsSummarizer (unchanged except formatting)
    // =======================================================================

    class DynamicsSummarizer : public featureExtraction
    {
    public:
        enum class DynamicsType
        {
            RMS,
            DynamicRange,
            CrestFactor,
            PeakLevel,
            Kurtosis
        };

        explicit DynamicsSummarizer(DynamicsType type)
            : dynamicsType(type)
        {
        }

        FeatureTensor extractFeatures(const AudioTensor &input) override
        {
            switch (dynamicsType)
            {
            case DynamicsType::RMS:
                return computeRMS(input);
            case DynamicsType::DynamicRange:
                return computeDynamicRange(input);
            case DynamicsType::CrestFactor:
                return computeCrestFactor(input);
            case DynamicsType::PeakLevel:
                return computePeakLevel(input);
            case DynamicsType::Kurtosis:
                return computeKurtosis(input);
            default:
                return FeatureTensor();
            }
        }

    private:
        float rms_db = 0.0f;
        float dynamic_range_db = 0.0f;
        float crest_factor_db = 0.0f;
        float peak_level_db = 0.0f;
        float kurtosis_db = 0.0f;

        DynamicsType dynamicsType;

        FeatureTensor computeRMS(const AudioTensor &input)
        {
            rms_db = 0.0f;
            for (int i = 0; i < input.numSamples; ++i)
                rms_db += input.data(i) * input.data(i);

            rms_db /= static_cast<float>(input.numSamples);
            rms_db = std::sqrt(rms_db);
            rms_db = 20.0f * std::log10(rms_db);

            FeatureTensor result;
            result.data.resize(1, 1);
            result.data(0, 0) = rms_db;
            result.numSamples = 1;
            result.features = 1;
            return result;
        }

        FeatureTensor computeDynamicRange(const AudioTensor &input)
        {
            float max_db = -std::numeric_limits<float>::max();
            float min_db = std::numeric_limits<float>::max();

            for (int i = 0; i < input.numSamples; ++i)
            {
                float v = std::max(std::abs(input.data(i)), 1e-12f);
                float sample_db = 20.0f * std::log10(v);
                max_db = std::max(max_db, sample_db);
                min_db = std::min(min_db, sample_db);
            }

            dynamic_range_db = max_db - min_db;

            FeatureTensor result;
            result.data.resize(1, 1);
            result.data(0, 0) = dynamic_range_db;
            result.numSamples = 1;
            result.features = 1;
            return result;
        }

        FeatureTensor computeCrestFactor(const AudioTensor &input)
        {
            float max_db = -std::numeric_limits<float>::max();
            for (int i = 0; i < input.numSamples; ++i)
            {
                float sample_db = 20.0f * std::log10(std::abs(input.data(i)));
                max_db = std::max(max_db, sample_db);
            }

            crest_factor_db = max_db - rms_db;

            FeatureTensor result;
            result.data.resize(1, 1);
            result.data(0, 0) = crest_factor_db;
            result.numSamples = 1;
            result.features = 1;
            return result;
        }

        FeatureTensor computePeakLevel(const AudioTensor &input)
        {
            float max_db = -std::numeric_limits<float>::max();
            for (int i = 0; i < input.numSamples; ++i)
            {
                float sample_db = 20.0f * std::log10(std::abs(input.data(i)));
                max_db = std::max(max_db, sample_db);
            }

            peak_level_db = max_db;

            FeatureTensor result;
            result.data.resize(1, 1);
            result.data(0, 0) = peak_level_db;
            result.numSamples = 1;
            result.features = 1;
            return result;
        }

        FeatureTensor computeKurtosis(const AudioTensor &input)
        {
            float kurtosis = 0.0f;
            for (int i = 0; i < input.numSamples; ++i)
                kurtosis += std::pow(input.data(i), 4);

            kurtosis /= static_cast<float>(input.numSamples);
            kurtosis -= 3.0f;
            kurtosis_db = kurtosis;

            FeatureTensor result;
            result.data.resize(1, 1);
            result.data(0, 0) = kurtosis_db;
            result.numSamples = 1;
            result.features = 1;
            return result;
        }
    };

} // namespace ChromaFlow
