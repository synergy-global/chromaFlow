#pragma once

// ==================================================================================
// Copyright (c) 2025 Synergy DSP
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =================================================================================


#include <cassert>
#include <cstring>
#include <vector> 
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> 
#endif
// TODO: Add other FFT implementations SSE, AVX, AVX2, AVX512
#if !defined(__APPLE__)
// TODO: Add other FFT implementations SSE, AVX, AVX2, AVX512
#endif

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