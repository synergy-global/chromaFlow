#pragma once

#include "ChromaBaseClasses.h"

#include "ChromaFFT.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace ChromaFlow
{

class FeatureExtractor : public featureExtraction
{
public:
    // Describes how columns in FeatureTensor are laid out
    struct Layout
    {
        int numMfcc       = 0;   // number of MFCC coefficients
        int mfccOffset    = -1;  // starting column index of MFCCs (usually 0 if used)

        int centroidIdx   = -1;
        int rolloffIdx    = -1;
        int bandwidthIdx  = -1;
        int energyIdx     = -1;
        int brightnessIdx = -1;
        int flatnessIdx   = -1;
        int zcrIdx        = -1;
    };

    FeatureExtractor (int sampleRate,
                      int nFft = 512,
                      int numMfcc = 13,
                      const std::vector<std::string>& featureNames = {
                          "mfcc",
                          "spectral_centroid",
                          "spectral_rolloff",
                          "spectral_bandwidth",
                          "spectral_energy",
                          "spectral_brightness",
                          "spectral_flatness",
                          "zcr" })
        : sample_rate (sampleRate),
          n_fft (std::max (8, nFft)),
          num_mfcc (std::max (1, numMfcc)),
          brightness_cutoff_hz (1500.0f),
          requested (featureNames.begin(), featureNames.end()),
          mel_n_mels (40)
    {
        // --- init FFT + static buffers (non-RT) ---
        n_fft = nearestPowerOfTwo (n_fft);

        fft.init (static_cast<size_t> (n_fft));
        complex_bins = static_cast<int> (fft.ComplexSize (static_cast<size_t> (n_fft)));

        ensureMelFilterbank();
        ensureDctMatrix();

        // Pre-allocate RT-safe buffers
        re_buffer.resize   (static_cast<size_t> (complex_bins));
        im_buffer.resize   (static_cast<size_t> (complex_bins));
        power_spectrum_buffer.resize      (complex_bins);
        magnitude_spectrum_buffer.resize  (complex_bins);
        freqs_buffer.resize               (complex_bins);
        cumsum_buffer.resize              (complex_bins);

        // Pre-calc bin freqs
        for (int k = 0; k < complex_bins; ++k)
        {
            freqs_buffer[k] = static_cast<float> (k)
                            * static_cast<float> (sample_rate)
                            / static_cast<float> (n_fft);
        }
    }

    // Column layout for a FeatureTensor produced by this extractor
    Layout getLayout() const
    {
        Layout L;
        int col = 0;

        // MFCC block first (if enabled)
        if (requested.count ("mfcc") > 0)
        {
            L.mfccOffset = col;
            L.numMfcc    = num_mfcc;
            col += num_mfcc;
        }

        // Scalars in the exact order we push in extractFeatures()
        if (requested.count ("spectral_centroid") > 0)
            L.centroidIdx = col++;

        if (requested.count ("spectral_rolloff") > 0)
            L.rolloffIdx = col++;

        if (requested.count ("spectral_bandwidth") > 0)
            L.bandwidthIdx = col++;

        if (requested.count ("spectral_energy") > 0)
            L.energyIdx = col++;

        if (requested.count ("spectral_brightness") > 0)
            L.brightnessIdx = col++;

        if (requested.count ("spectral_flatness") > 0)
            L.flatnessIdx = col++;

        if (requested.count ("zcr") > 0)
            L.zcrIdx = col++;

        return L;
    }

    // Extract features from a raw audio block
    FeatureTensor extractFeatures (const AudioTensor& input) override
    {
        // Mono block from input.data
        const int inSize = static_cast<int> (input.data.size());
        Eigen::VectorXf block;
        if (inSize < n_fft)
        {
            block = Eigen::VectorXf::Zero (n_fft);
            block.head (inSize) = input.data;
        }
        else
        {
            block = input.data.head (n_fft);
        }

        // ----------------- ZCR (time-domain) -----------------
        float zcr = 0.0f;
        if (requested.count ("zcr") > 0)
        {
            int crossings = 0;
            const int N = static_cast<int> (block.size());
            if (N > 1)
            {
                float prev = block[0];
                for (int n = 1; n < N; ++n)
                {
                    float curr = block[n];
                    if ((prev >= 0.0f && curr < 0.0f) ||
                        (prev < 0.0f  && curr >= 0.0f))
                    {
                        ++crossings;
                    }
                    prev = curr;
                }
                zcr = static_cast<float> (crossings)
                    / static_cast<float> (N - 1);
            }
        }

        // ----------------- FFT + spectral stuff -----------------
        fft.fft (block.data(), re_buffer.data(), im_buffer.data());

        // Power / mag spectra
        for (int k = 0; k < complex_bins; ++k)
        {
            const float rr = re_buffer[static_cast<size_t> (k)];
            const float ii = im_buffer[static_cast<size_t> (k)];
            const float p  = rr * rr + ii * ii;
            power_spectrum_buffer[k]     = p;
            magnitude_spectrum_buffer[k] = std::sqrt (std::max (0.0f, p));
        }

        const float spec_sum       = magnitude_spectrum_buffer.sum() + 1e-6f;
        const float power_spec_sum = power_spectrum_buffer.sum() + 1e-6f;

        std::vector<float> out;
        out.reserve (static_cast<size_t> (num_mfcc + 7)); // MFCCs + up to 7 scalars

        // ----------------- MFCCs -----------------
        if (requested.count ("mfcc") > 0)
        {
            Eigen::VectorXf mel_energies = (mel_filterbank * power_spectrum_buffer).eval();
            Eigen::VectorXf log_mel      = (mel_energies.array() + 1e-6f).log().matrix().eval();
            Eigen::VectorXf mfcc         = (dct_matrix * log_mel).eval(); // size num_mfcc

            for (int i = 0; i < mfcc.size(); ++i)
                out.push_back (mfcc[i]);
        }

        // ----------------- Scalar spectral features -----------------
        float brightness = 0.0f;
        if (requested.count ("spectral_brightness") > 0)
        {
            const int brightness_bin = static_cast<int> (
                brightness_cutoff_hz / (static_cast<float> (sample_rate) / static_cast<float> (n_fft)));
            const int start = std::min (std::max (0, brightness_bin), complex_bins - 1);
            const float high_energy =
                power_spectrum_buffer.segment (start, complex_bins - start).sum();
            brightness = high_energy / power_spec_sum;
        }

        float centroid = 0.0f;
        if (requested.count ("spectral_centroid") > 0
            || requested.count ("spectral_bandwidth") > 0)
        {
            centroid = static_cast<float> (
                (freqs_buffer.array() * magnitude_spectrum_buffer.array()).sum()) / spec_sum;
        }

        float bandwidth = 0.0f;
        if (requested.count ("spectral_bandwidth") > 0)
        {
            Eigen::ArrayXf diff2 = (freqs_buffer.array() - centroid).square();
            bandwidth = std::sqrt (static_cast<float> (
                (diff2 * magnitude_spectrum_buffer.array()).sum()) / spec_sum);
        }

        float rolloff = 0.0f;
        if (requested.count ("spectral_rolloff") > 0)
        {
            float acc = 0.0f;
            for (int k = 0; k < complex_bins; ++k)
            {
                acc += magnitude_spectrum_buffer[k];
                cumsum_buffer[k] = acc;
            }

            const float target = 0.85f * cumsum_buffer[complex_bins - 1];
            int idx = 0;
            while (idx < complex_bins && cumsum_buffer[idx] < target)
                ++idx;
            if (idx >= complex_bins)
                idx = complex_bins - 1;
            rolloff = freqs_buffer[idx];
        }

        float energy = 0.0f;
        if (requested.count ("spectral_energy") > 0)
            energy = power_spec_sum;

        // spectral flatness (power spectrum)
        float flatness = 0.0f;
        if (requested.count ("spectral_flatness") > 0)
        {
            const int N = complex_bins;
            if (N > 0)
            {
                auto p = power_spectrum_buffer.array() + 1e-12f;
                const float geoMean   = std::exp (p.log().sum() / static_cast<float> (N));
                const float arithMean = p.mean();
                flatness = geoMean / (arithMean + 1e-12f);
            }
        }

        // Append in fixed order (must match getLayout)
        if (requested.count ("spectral_centroid") > 0)
            out.push_back (centroid);
        if (requested.count ("spectral_rolloff") > 0)
            out.push_back (rolloff);
        if (requested.count ("spectral_bandwidth") > 0)
            out.push_back (bandwidth);
        if (requested.count ("spectral_energy") > 0)
            out.push_back (energy);
        if (requested.count ("spectral_brightness") > 0)
            out.push_back (brightness);
        if (requested.count ("spectral_flatness") > 0)
            out.push_back (flatness);
        if (requested.count ("zcr") > 0)
            out.push_back (zcr);

        // Pack into FeatureTensor
        FeatureTensor featuresTensor;
        const int F = static_cast<int> (out.size());
        featuresTensor.data.resize (1, F);
        for (int i = 0; i < F; ++i)
            featuresTensor.data (0, i) = out[static_cast<size_t> (i)];

        featuresTensor.numSamples = 1;
        featuresTensor.features   = F;
        return featuresTensor;
    }

private:
    int sample_rate;
    int n_fft;
    int num_mfcc;
    float brightness_cutoff_hz;
    std::unordered_set<std::string> requested;
    int mel_n_mels;
    int complex_bins;
    ChromaFFT::AppleAccelerateFFT fft;
    Eigen::MatrixXf mel_filterbank; // [mel_n_mels x complex_bins]
    Eigen::MatrixXf dct_matrix;     // [num_mfcc   x mel_n_mels]

    std::vector<float> re_buffer;
    std::vector<float> im_buffer;
    Eigen::VectorXf    power_spectrum_buffer;
    Eigen::VectorXf    magnitude_spectrum_buffer;
    Eigen::VectorXf    freqs_buffer;
    Eigen::VectorXf    cumsum_buffer;

    static int nearestPowerOfTwo (int n)
    {
        int p = 1;
        while ((p << 1) <= n)
            p <<= 1;
        return p;
    }

    void ensureMelFilterbank()
    {
        if (mel_filterbank.rows() == mel_n_mels
            && mel_filterbank.cols() == complex_bins)
            return;

        const float low_freq_mel  = 0.0f;
        const float high_freq_mel =
            2595.0f * std::log10 (1.0f + (static_cast<float> (sample_rate) / 2.0f) / 700.0f);

        Eigen::VectorXf mel_points =
            Eigen::VectorXf::LinSpaced (mel_n_mels + 2, low_freq_mel, high_freq_mel);

        Eigen::VectorXf hz_points (mel_points.size());
        for (int i = 0; i < mel_points.size(); ++i)
        {
            hz_points[i] =
                700.0f * (std::pow (10.0f, mel_points[i] / 2595.0f) - 1.0f);
        }

        std::vector<int> bin_points (static_cast<size_t> (mel_points.size()));
        for (int i = 0; i < hz_points.size(); ++i)
        {
            const float binF = (static_cast<float> (n_fft) + 1.0f)
                             * hz_points[i] / static_cast<float> (sample_rate);
            int bin = static_cast<int> (std::floor (binF));
            bin = std::max (0, std::min (bin, complex_bins - 1));
            bin_points[static_cast<size_t> (i)] = bin;
        }

        mel_filterbank.resize (mel_n_mels, complex_bins);
        mel_filterbank.setZero();
        for (int m = 1; m <= mel_n_mels; ++m)
        {
            const int f_m_minus = bin_points[static_cast<size_t> (m - 1)];
            const int f_m       = bin_points[static_cast<size_t> (m)];
            const int f_m_plus  = bin_points[static_cast<size_t> (m + 1)];
            if (f_m_minus >= f_m_plus)
                continue;

            for (int k = f_m_minus; k < f_m; ++k)
            {
                float denom = std::max (1, f_m - f_m_minus);
                mel_filterbank (m - 1, k) =
                    (static_cast<float> (k - f_m_minus) / static_cast<float> (denom));
            }

            for (int k = f_m; k < f_m_plus; ++k)
            {
                float denom = std::max (1, f_m_plus - f_m);
                mel_filterbank (m - 1, k) =
                    (static_cast<float> (f_m_plus - k) / static_cast<float> (denom));
            }
        }
    }

    void ensureDctMatrix()
    {
        if (dct_matrix.rows() == num_mfcc
            && dct_matrix.cols() == mel_n_mels)
            return;

        dct_matrix = Eigen::MatrixXf (num_mfcc, mel_n_mels);
        const float N      = static_cast<float> (mel_n_mels);
        const float scale0 = std::sqrt (1.0f / (4.0f * N));
        const float scale  = std::sqrt (1.0f / (2.0f * N));
        const float pi     = 3.14159265358979323846f;

        for (int k = 0; k < num_mfcc; ++k)
        {
            for (int n = 0; n < mel_n_mels; ++n)
            {
                const float angle = pi * (static_cast<float> (n) + 0.5f)
                                  * static_cast<float> (k) / N;
                float s = std::cos (angle);
                dct_matrix (k, n) = (k == 0 ? (2.0f * scale0) : (2.0f * scale)) * s;
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

    explicit DynamicsSummarizer (DynamicsType type)
        : dynamicsType (type)
    {}

    FeatureTensor extractFeatures (const AudioTensor& input) override
    {
        switch (dynamicsType)
        {
            case DynamicsType::RMS:          return computeRMS          (input);
            case DynamicsType::DynamicRange: return computeDynamicRange (input);
            case DynamicsType::CrestFactor:  return computeCrestFactor  (input);
            case DynamicsType::PeakLevel:    return computePeakLevel    (input);
            case DynamicsType::Kurtosis:     return computeKurtosis     (input);
            default:                         return FeatureTensor();
        }
    }

private:
    float rms_db          = 0.0f;
    float dynamic_range_db = 0.0f;
    float crest_factor_db  = 0.0f;
    float peak_level_db    = 0.0f;
    float kurtosis_db      = 0.0f;

    DynamicsType dynamicsType;

    FeatureTensor computeRMS (const AudioTensor& input)
    {
        rms_db = 0.0f;
        for (int i = 0; i < input.numSamples; ++i)
            rms_db += input.data (i) * input.data (i);

        rms_db /= static_cast<float> (input.numSamples);
        rms_db  = std::sqrt (rms_db);
        rms_db  = 20.0f * std::log10 (rms_db);

        FeatureTensor result;
        result.data.resize (1, 1);
        result.data (0, 0) = rms_db;
        result.numSamples  = 1;
        result.features    = 1;
        return result;
    }

    FeatureTensor computeDynamicRange (const AudioTensor& input)
    {
        float max_db = -std::numeric_limits<float>::max();
        float min_db =  std::numeric_limits<float>::max();

        for (int i = 0; i < input.numSamples; ++i)
        {
            float sample_db = 20.0f * std::log10 (std::abs (input.data (i)));
            max_db = std::max (max_db, sample_db);
            min_db = std::min (min_db, sample_db);
        }

        dynamic_range_db = max_db - min_db;

        FeatureTensor result;
        result.data.resize (1, 1);
        result.data (0, 0) = dynamic_range_db;
        result.numSamples  = 1;
        result.features    = 1;
        return result;
    }

    FeatureTensor computeCrestFactor (const AudioTensor& input)
    {
        float max_db = -std::numeric_limits<float>::max();
        for (int i = 0; i < input.numSamples; ++i)
        {
            float sample_db = 20.0f * std::log10 (std::abs (input.data (i)));
            max_db = std::max (max_db, sample_db);
        }

        crest_factor_db = max_db - rms_db;

        FeatureTensor result;
        result.data.resize (1, 1);
        result.data (0, 0) = crest_factor_db;
        result.numSamples  = 1;
        result.features    = 1;
        return result;
    }

    FeatureTensor computePeakLevel (const AudioTensor& input)
    {
        float max_db = -std::numeric_limits<float>::max();
        for (int i = 0; i < input.numSamples; ++i)
        {
            float sample_db = 20.0f * std::log10 (std::abs (input.data (i)));
            max_db = std::max (max_db, sample_db);
        }

        peak_level_db = max_db;

        FeatureTensor result;
        result.data.resize (1, 1);
        result.data (0, 0) = peak_level_db;
        result.numSamples  = 1;
        result.features    = 1;
        return result;
    }

    FeatureTensor computeKurtosis (const AudioTensor& input)
    {
        float kurtosis = 0.0f;
        for (int i = 0; i < input.numSamples; ++i)
            kurtosis += std::pow (input.data (i), 4);

        kurtosis /= static_cast<float> (input.numSamples);
        kurtosis -= 3.0f;
        kurtosis_db = kurtosis;

        FeatureTensor result;
        result.data.resize (1, 1);
        result.data (0, 0) = kurtosis_db;
        result.numSamples  = 1;
        result.features    = 1;
        return result;
    }
};

} // namespace ChromaFlow
