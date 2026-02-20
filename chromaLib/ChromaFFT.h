#pragma once


#include <cassert>
#include <cstring>
#include <vector> 
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> 
#endif 
#if !defined(__APPLE__)

class Radix2FFTImpl : public ChromaFFTImpl {
public:
    Radix2FFTImpl()
        : _size(0), _log2(0)
    {}

    void init(size_t size) override
    {
        assert(IsPowerOf2(size));
        _size = size;
        _log2 = 0;
        while ((static_cast<size_t>(1) << _log2) < _size)
            ++_log2;

        _bitRev.resize(_size);
        _twRe.resize(_size / 2);
        _twIm.resize(_size / 2);

        computeBitReverse();
        computeTwiddles();
    }

    void fft(const float* data, float* re, float* im) override
    {
        // Copy input
        for (size_t i = 0; i < _size; ++i)
        {
            re[i] = data[i];
            im[i] = 0.0f;
        }

        bitReverseReorder(re, im);

        for (size_t stage = 1; stage <= _log2; ++stage)
        {
            size_t m = 1 << stage;
            size_t m2 = m >> 1;
            size_t step = _size / m;

            for (size_t k = 0; k < _size; k += m)
            {
                for (size_t j = 0; j < m2; ++j)
                {
                    size_t tw = j * step;

                    float wr = _twRe[tw];
                    float wi = _twIm[tw];

                    size_t t = k + j;
                    size_t u = t + m2;

                    float tr = wr * re[u] - wi * im[u];
                    float ti = wr * im[u] + wi * re[u];

                    float ur = re[t];
                    float ui = im[t];

                    re[t] = ur + tr;
                    im[t] = ui + ti;

                    re[u] = ur - tr;
                    im[u] = ui - ti;
                }
            }
        }
    }

    void ifft(float* data, const float* reIn, const float* imIn) override
    {
        // Copy input
        for (size_t i = 0; i < _size; ++i)
        {
            _reTmp[i] = reIn[i];
            _imTmp[i] = imIn[i];
        }

        // Conjugate
        for (size_t i = 0; i < _size; ++i)
            _imTmp[i] = -_imTmp[i];

        // Forward FFT
        fftInternal(_reTmp.data(), _imTmp.data());

        // Conjugate + scale
        const float scale = 1.0f / static_cast<float>(_size);
        for (size_t i = 0; i < _size; ++i)
            data[i] = _reTmp[i] * scale;
    }

    size_t ComplexSize(size_t size) override
    {
        return size / 2 + 1;
    }

private:
    size_t _size;
    size_t _log2;

    std::vector<size_t> _bitRev;
    std::vector<float> _twRe;
    std::vector<float> _twIm;

    std::vector<float> _reTmp;
    std::vector<float> _imTmp;

    void computeBitReverse()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            size_t x = i;
            size_t r = 0;
            for (size_t b = 0; b < _log2; ++b)
            {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            _bitRev[i] = r;
        }
    }

    void computeTwiddles()
    {
        constexpr float twoPi = 6.28318530717958647692f;

        for (size_t k = 0; k < _size / 2; ++k)
        {
            float angle = -twoPi * k / static_cast<float>(_size);
            _twRe[k] = std::cos(angle);
            _twIm[k] = std::sin(angle);
        }

        _reTmp.resize(_size);
        _imTmp.resize(_size);
    }

    void bitReverseReorder(float* re, float* im)
    {
        for (size_t i = 0; i < _size; ++i)
        {
            size_t j = _bitRev[i];
            if (j > i)
            {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
    }

    void fftInternal(float* re, float* im)
    {
        bitReverseReorder(re, im);

        for (size_t stage = 1; stage <= _log2; ++stage)
        {
            size_t m = 1 << stage;
            size_t m2 = m >> 1;
            size_t step = _size / m;

            for (size_t k = 0; k < _size; k += m)
            {
                for (size_t j = 0; j < m2; ++j)
                {
                    size_t tw = j * step;

                    float wr = _twRe[tw];
                    float wi = _twIm[tw];

                    size_t t = k + j;
                    size_t u = t + m2;

                    float tr = wr * re[u] - wi * im[u];
                    float ti = wr * im[u] + wi * re[u];

                    float ur = re[t];
                    float ui = im[t];

                    re[t] = ur + tr;
                    im[t] = ui + ti;

                    re[u] = ur - tr;
                    im[u] = ui - ti;
                }
            }
        }
    }
};

#endif

// TODO: Add documentation for FFT implementation
namespace ChromaFFT {
 

class ChromaFFTImpl {
public:
  ChromaFFTImpl() = default;
  ChromaFFTImpl(const ChromaFFTImpl&) = delete;
  ChromaFFTImpl& operator=(const ChromaFFTImpl&) = delete;
  virtual ~ChromaFFTImpl() = default;
  virtual void init(size_t size) = 0;
  virtual void fft(const float* data, float* re, float* im) = 0;
  virtual void ifft(float* data, const float* re, const float* im) = 0;
  virtual size_t ComplexSize(size_t size) = 0;
};

constexpr bool IsPowerOf2(size_t val) {
  return (val == 1 || (val & (val - 1)) == 0);
}
class AppleAccelerateFFT : public ChromaFFTImpl {
public:
  AppleAccelerateFFT()
      : ChromaFFTImpl(), _size(0), _powerOf2(0), _fftSetup(0), _re(),
        _im() {}

  AppleAccelerateFFT(const AppleAccelerateFFT&) = delete;
  AppleAccelerateFFT& operator=(const AppleAccelerateFFT&) = delete;

  ~AppleAccelerateFFT() override { init(0); }

  void init(size_t size) override {
    if (_fftSetup) {
      vDSP_destroy_fftsetup(_fftSetup);
      _size = 0;
      _powerOf2 = 0;
      _fftSetup = 0;
      _re.clear();
      _im.clear();
    }

    if (size > 0) {
      _size = size;
      _powerOf2 = 0;
      while ((static_cast<size_t>(1) << _powerOf2) < _size) {
        ++_powerOf2;
      }
      _fftSetup = vDSP_create_fftsetup(_powerOf2, FFT_RADIX2);
      _re.resize(_size / 2);
      _im.resize(_size / 2);
    }
  }

  void fft(const float* data, float* re, float* im) override {
    const size_t size2 = _size / 2;
    DSPSplitComplex splitComplex;
    splitComplex.realp = re;
    splitComplex.imagp = im;
    vDSP_ctoz(reinterpret_cast<const COMPLEX*>(data), 2, &splitComplex, 1,
              size2);
    vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_FORWARD);
    const float factor = 0.5f;
    vDSP_vsmul(re, 1, &factor, re, 1, size2);
    vDSP_vsmul(im, 1, &factor, im, 1, size2);
    re[size2] = im[0];
    im[0] = 0.0f;
    im[size2] = 0.0f;
  }

  void ifft(float* data, const float* re, const float* im) override {
    const size_t size2 = _size / 2;
    std::memcpy(_re.data(), re, size2 * sizeof(float));
    std::memcpy(_im.data(), im, size2 * sizeof(float));
    _im[0] = re[size2];
    DSPSplitComplex splitComplex;
    splitComplex.realp = _re.data();
    splitComplex.imagp = _im.data();
    vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_INVERSE);
    vDSP_ztoc(&splitComplex, 1, reinterpret_cast<COMPLEX*>(data), 2, size2);
    const float factor = 1.0f / static_cast<float>(_size);
    vDSP_vsmul(data, 1, &factor, data, 1, _size);
  }

  size_t ComplexSize(size_t size) override {
    return size / 2 + 1;
  }
private:
  size_t _size;
  size_t _powerOf2;
  FFTSetup _fftSetup;
  std::vector<float> _re;
  std::vector<float> _im;
};

// TODO: Add other FFT implementations and client 



} // namespace ChromaFFT