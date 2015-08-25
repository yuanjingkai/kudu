/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * Copyright (c) 2014 Daniel Lemire
 *
 * Some of this code is cribbed from https://github.com/lemire/FastPFor,
 * but modified to add the computation of frame of reference.
 */
#ifndef BITPACKING_H
#define BITPACKING_H
#include <smmintrin.h>
#include <immintrin.h>

namespace kudu {
namespace coding {

static void SIMD_nullunpacker32(const __m128i * __restrict__, uint32_t * __restrict__ _out,
                                const uint32_t frame_of_reference) {
  for (uint32_t i = 0; i < 128; i++) {
    *_out++ = frame_of_reference;
  }
}

static void __SIMD_fastunpack1_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg1 = _mm_loadu_si128(in);
  __m128i       InReg2 = InReg1;
  __m128i       OutReg1, OutReg2, OutReg3, OutReg4;
  const __m128i mask = _mm_set1_epi32(1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  unsigned      shift = 0;

  for (unsigned i = 0; i < 8; ++i) {
    OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, shift++), mask);
    OutReg1 = _mm_add_epi32(OutReg1, add);
    OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, shift++), mask);
    OutReg2 = _mm_add_epi32(OutReg2, add);
    OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, shift++), mask);
    OutReg3 = _mm_add_epi32(OutReg3, add);
    OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, shift++), mask);
    OutReg4 = _mm_add_epi32(OutReg4, add);
    _mm_storeu_si128(out++, OutReg1);
    _mm_storeu_si128(out++, OutReg2);
    _mm_storeu_si128(out++, OutReg3);
    _mm_storeu_si128(out++, OutReg4);
  }
}

static void __SIMD_fastunpack2_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 2) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack3_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 3) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 27), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 25), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack4_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 4) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack5_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 5) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 25), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack6_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 6) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack7_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 7) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack8_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 8) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack9_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                  const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 9) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack10_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 10) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack11_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 11) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack12_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 12) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack13_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 13) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack14_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 14) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack15_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 15) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack16_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 16) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack17_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 17) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack18_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 18) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack19_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 19) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack20_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 20) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack21_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 21) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack22_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 22) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack23_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 23) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 21), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack24_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 24) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack25_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 25) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 23), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 9);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 21), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack26_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 26) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack27_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 27) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 7);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 9);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 26), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 21), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 23), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 25), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack28_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 28) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack29_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 29) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 26), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 23), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 5);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 28), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 25), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 7);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 27), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 21), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 9);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack30_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 30) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 28), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 26), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 2);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 28), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 26), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack31_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i       *out = reinterpret_cast < __m128i * > (_out);
  __m128i       InReg = _mm_loadu_si128(in);
  __m128i       OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 31) - 1);
  float         min_delta = bit_cast<float> (frame_of_reference);
  __m128i       add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  OutReg = _mm_and_si128(InReg, mask);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 30), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 29), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 28), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 27), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 26), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 25), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 24), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 23), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 23);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 22), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 22);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 21), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 21);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 20), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 20);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 19), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 19);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 18), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 18);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 17), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 17);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 16), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 16);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 15), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 15);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 14), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 14);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 13), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 13);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 12), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 12);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 11), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 11);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 10), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 10);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 9), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 9);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 8), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 8);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 7), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 7);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 6), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 6);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 5), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 5);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 4), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 4);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 3), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 3);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 2), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 2);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 1), mask));
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_add_epi32(OutReg, add);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack32_32(const __m128i * __restrict__ in, uint32_t * __restrict__ _out,
                                   const uint32_t frame_of_reference) {
  __m128i *out = reinterpret_cast < __m128i * > (_out);
  __m128i OutReg;
  float   min_delta = bit_cast<float> (frame_of_reference);
  __m128i add = (__m128i) _mm_set_ps(min_delta, min_delta, min_delta, min_delta);

  for (uint32_t outer = 0; outer < 32; ++outer) {
    OutReg = _mm_loadu_si128(in++);
    OutReg = _mm_add_epi32(OutReg, add);
    _mm_storeu_si128(out++, OutReg);
  }
}

static void usimdunpack(const __m128i * __restrict__ in,
                        uint32_t * __restrict__ out,
                        const uint32_t bit,
                        const uint32_t frame_of_reference) {
  switch (bit) {
    case 0:   SIMD_nullunpacker32(in, out, frame_of_reference); return;
    case 1:   __SIMD_fastunpack1_32(in, out, frame_of_reference); return;
    case 2:   __SIMD_fastunpack2_32(in, out, frame_of_reference); return;
    case 3:   __SIMD_fastunpack3_32(in, out, frame_of_reference); return;
    case 4:   __SIMD_fastunpack4_32(in, out, frame_of_reference); return;
    case 5:   __SIMD_fastunpack5_32(in, out, frame_of_reference); return;
    case 6:   __SIMD_fastunpack6_32(in, out, frame_of_reference); return;
    case 7:   __SIMD_fastunpack7_32(in, out, frame_of_reference); return;
    case 8:   __SIMD_fastunpack8_32(in, out, frame_of_reference); return;
    case 9:   __SIMD_fastunpack9_32(in, out, frame_of_reference); return;
    case 10:  __SIMD_fastunpack10_32(in, out, frame_of_reference); return;
    case 11:  __SIMD_fastunpack11_32(in, out, frame_of_reference); return;
    case 12:  __SIMD_fastunpack12_32(in, out, frame_of_reference); return;
    case 13:  __SIMD_fastunpack13_32(in, out, frame_of_reference); return;
    case 14:  __SIMD_fastunpack14_32(in, out, frame_of_reference); return;
    case 15:  __SIMD_fastunpack15_32(in, out, frame_of_reference); return;
    case 16:  __SIMD_fastunpack16_32(in, out, frame_of_reference); return;
    case 17:  __SIMD_fastunpack17_32(in, out, frame_of_reference); return;
    case 18:  __SIMD_fastunpack18_32(in, out, frame_of_reference); return;
    case 19:  __SIMD_fastunpack19_32(in, out, frame_of_reference); return;
    case 20:  __SIMD_fastunpack20_32(in, out, frame_of_reference); return;
    case 21:  __SIMD_fastunpack21_32(in, out, frame_of_reference); return;
    case 22:  __SIMD_fastunpack22_32(in, out, frame_of_reference); return;
    case 23:  __SIMD_fastunpack23_32(in, out, frame_of_reference); return;
    case 24:  __SIMD_fastunpack24_32(in, out, frame_of_reference); return;
    case 25:  __SIMD_fastunpack25_32(in, out, frame_of_reference); return;
    case 26:  __SIMD_fastunpack26_32(in, out, frame_of_reference); return;
    case 27:  __SIMD_fastunpack27_32(in, out, frame_of_reference); return;
    case 28:  __SIMD_fastunpack28_32(in, out, frame_of_reference); return;
    case 29:  __SIMD_fastunpack29_32(in, out, frame_of_reference); return;
    case 30:  __SIMD_fastunpack30_32(in, out, frame_of_reference); return;
    case 31:  __SIMD_fastunpack31_32(in, out, frame_of_reference); return;
    case 32:  __SIMD_fastunpack32_32(in, out, frame_of_reference); return;
    default:  break;
  }
}
} // namespace coding
} // namespace kudu

#endif // BITPACKING_H
