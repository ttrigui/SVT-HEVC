/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbMemory_AVX2_h
#define EbMemory_AVX2_h

#include "EbDefinitions.h"
#include "immintrin.h"
//#include "EbTypes.h"
//#include "../VPX/vpx_dsp_rtcd.h"

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
#endif

#ifndef _mm256_setr_m128i
#define _mm256_setr_m128i(/* __m128i */ lo, /* __m128i */ hi) \
    _mm256_set_m128i((hi), (lo))
#endif

static inline __m256i load8bit_4x4_avx2(const EB_U8 *const src,
    const EB_U32 stride)
{
    __m128i src01, src23;
    src01 = _mm_cvtsi32_si128(*(EB_S32*)(src + 0 * stride));
    src01 = _mm_insert_epi32(src01, *(EB_S32 *)(src + 1 * stride), 1);
    src23 = _mm_cvtsi32_si128(*(EB_S32*)(src + 2 * stride));
    src23 = _mm_insert_epi32(src23, *(EB_S32 *)(src + 3 * stride), 1);
    return _mm256_setr_m128i(src01, src23);
}

static inline __m128i load64bit_2_sse2(const void *const src,
    const EB_U32 stride)
{
    const EB_U8 *const s = (const EB_U8 *)src;
    const __m128i src0 = _mm_loadl_epi64((__m128i *)(s + 0 * stride));
    const __m128i src1 = _mm_loadl_epi64((__m128i *)(s + 1 * stride));
    return _mm_unpacklo_epi64(src0, src1);
}

static inline __m256i load8bit_8x4_avx2(const EB_U8 *const src,
    const EB_U32 stride)
{
    const __m128i src01 = load64bit_2_sse2(src + 0 * stride, stride);
    const __m128i src23 = load64bit_2_sse2(src + 2 * stride, stride);
    return _mm256_setr_m128i(src01, src23);
}

static inline __m256i load8bit_16x2_avx2(const EB_U8 *const src,
    const EB_U32 stride)
{
    const __m128i src0 = _mm_load_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_load_si128((__m128i *)(src + 1 * stride));
    return _mm256_setr_m128i(src0, src1);
}

static inline __m256i load8bit_16x2_unaligned_avx2(const EB_U8 *const src,
    const EB_U32 stride)
{
    const __m128i src0 = _mm_loadu_si128((__m128i *)(src + 0 * stride));
    const __m128i src1 = _mm_loadu_si128((__m128i *)(src + 1 * stride));
    return _mm256_setr_m128i(src0, src1);
}

static inline __m256i load16bit_signed_4x4_avx2(const EB_S16 *const src,
    const EB_U32 stride)
{
    const __m128i src01 = load64bit_2_sse2(src + 0 * stride, sizeof(*src) * stride);
    const __m128i src23 = load64bit_2_sse2(src + 2 * stride, sizeof(*src) * stride);
    return _mm256_setr_m128i(src01, src23);
}

#endif // EbMemory_AVX2_h
