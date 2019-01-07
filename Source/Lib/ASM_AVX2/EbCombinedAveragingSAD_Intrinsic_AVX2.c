/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/


#include "EbCombinedAveragingSAD_Intrinsic_AVX2.h"
#include "EbMemory_AVX2.h"
#include "immintrin.h"

EB_U32 CombinedAveraging8xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y += 4) {
		const __m256i s = load8bit_8x4_avx2(src, srcStride);
		const __m256i r1 = load8bit_8x4_avx2(ref1, ref1Stride);
		const __m256i r2 = load8bit_8x4_avx2(ref2, ref2Stride);
		const __m256i avg = _mm256_avg_epu8(r1, r2);
		const __m256i sad = _mm256_sad_epu8(s, avg);
		sum = _mm256_add_epi32(sum, sad);
		src += srcStride << 2;
		ref1 += ref1Stride << 2;
		ref2 += ref2Stride << 2;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm256_extracti128_si256(sum, 1));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}

static inline __m256i CombinedAveragingSad16x2_AVX2(const EB_U8 *const src,
	const EB_U32 srcStride, const EB_U8 *const ref1, const EB_U32 ref1Stride,
	const EB_U8 *const ref2, const EB_U32 ref2Stride, const __m256i sum)
{
	const __m256i s = load8bit_16x2_unaligned_avx2(src, srcStride);
	const __m256i r1 = load8bit_16x2_unaligned_avx2(ref1, ref1Stride);
	const __m256i r2 = load8bit_16x2_unaligned_avx2(ref2, ref2Stride);
	const __m256i avg = _mm256_avg_epu8(r1, r2);
	const __m256i sad = _mm256_sad_epu8(s, avg);
	return _mm256_add_epi32(sum, sad);
}

EB_U32 CombinedAveraging16xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y += 2) {
		sum = CombinedAveragingSad16x2_AVX2(src, srcStride, ref1, ref1Stride,
			ref2, ref2Stride, sum);
		src += srcStride << 1;
		ref1 += ref1Stride << 1;
		ref2 += ref2Stride << 1;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm256_extracti128_si256(sum, 1));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}

static inline __m256i CombinedAveragingSad24_AVX2(const EB_U8 *const src,
	const EB_U8 *const ref1, const EB_U8 *const ref2, const __m256i sum)
{
	const __m256i s = _mm256_loadu_si256((__m256i*)src);
	const __m256i r1 = _mm256_loadu_si256((__m256i*)ref1);
	const __m256i r2 = _mm256_loadu_si256((__m256i*)ref2);
	const __m256i avg = _mm256_avg_epu8(r1, r2);
	const __m256i sad = _mm256_sad_epu8(s, avg);
	return _mm256_add_epi32(sum, sad);
}

EB_U32 CombinedAveraging24xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y += 2) {
		sum = CombinedAveragingSad24_AVX2(src + 0 * srcStride,
			ref1 + 0 * ref1Stride, ref2 + 0 * ref2Stride, sum);
		sum = CombinedAveragingSad24_AVX2(src + 1 * srcStride,
			ref1 + 1 * ref1Stride, ref2 + 1 * ref2Stride, sum);
		src += srcStride << 1;
		ref1 += ref1Stride << 1;
		ref2 += ref2Stride << 1;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm_slli_si128(_mm256_extracti128_si256(sum, 1), 8));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}

static inline __m256i CombinedAveragingSad32_AVX2(const EB_U8 *const src,
	const EB_U8 *const ref1, const EB_U8 *const ref2, const __m256i sum)
{
	const __m256i s = _mm256_loadu_si256((__m256i*)src);
	const __m256i r1 = _mm256_loadu_si256((__m256i*)ref1);
	const __m256i r2 = _mm256_loadu_si256((__m256i*)ref2);
	const __m256i avg = _mm256_avg_epu8(r1, r2);
	const __m256i sad = _mm256_sad_epu8(s, avg);
	return _mm256_add_epi32(sum, sad);
}

EB_U32 CombinedAveraging32xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y += 2) {
		sum = CombinedAveragingSad32_AVX2(src + 0 * srcStride,
			ref1 + 0 * ref1Stride, ref2 + 0 * ref2Stride, sum);
		sum = CombinedAveragingSad32_AVX2(src + 1 * srcStride,
			ref1 + 1 * ref1Stride, ref2 + 1 * ref2Stride, sum);
		src += srcStride << 1;
		ref1 += ref1Stride << 1;
		ref2 += ref2Stride << 1;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm256_extracti128_si256(sum, 1));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}

EB_U32 CombinedAveraging48xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y += 2) {
		sum = CombinedAveragingSad32_AVX2(src + 0 * srcStride,
			ref1 + 0 * ref1Stride, ref2 + 0 * ref2Stride, sum);
		sum = CombinedAveragingSad32_AVX2(src + 1 * srcStride,
			ref1 + 1 * ref1Stride, ref2 + 1 * ref2Stride, sum);
		sum = CombinedAveragingSad16x2_AVX2(src + 32, srcStride, ref1 + 32,
			ref1Stride, ref2 + 32, ref2Stride, sum);

		src += srcStride << 1;
		ref1 += ref1Stride << 1;
		ref2 += ref2Stride << 1;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm256_extracti128_si256(sum, 1));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}

EB_U32 CombinedAveraging64xMSAD_AVX2_INTRIN(
	EB_U8  *src,
	EB_U32  srcStride,
	EB_U8  *ref1,
	EB_U32  ref1Stride,
	EB_U8  *ref2,
	EB_U32  ref2Stride,
	EB_U32  height,
	EB_U32  width)
{
	__m256i sum = _mm256_setzero_si256();
	__m128i sad;
	EB_U32 y;
	(void)width;

	for (y = 0; y < height; y++) {
		sum = CombinedAveragingSad32_AVX2(src + 0x00,
			ref1 + 0x00, ref2 + 0x00, sum);
		sum = CombinedAveragingSad32_AVX2(src + 0x20,
			ref1 + 0x20, ref2 + 0x20, sum);
		src += srcStride;
		ref1 += ref1Stride;
		ref2 += ref2Stride;
	}

	sad = _mm_add_epi32(_mm256_castsi256_si128(sum),
		_mm256_extracti128_si256(sum, 1));
	sad = _mm_add_epi32(sad, _mm_srli_si128(sad, 8));

	return _mm_cvtsi128_si32(sad);
}
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

EB_U64 ComputeMean8x8_AVX2_INTRIN(
    EB_U8 *  inputSamples,      // input parameter, input samples Ptr
    EB_U32   inputStride,       // input parameter, input stride
    EB_U32   inputAreaWidth,    // input parameter, input area width
    EB_U32   inputAreaHeight)   // input parameter, input area height
{
	__m256i sum,sum2 ,xmm2, xmm1, sum1, xmm0 = _mm256_setzero_si256();
	__m128i  upper, lower, mean = _mm_setzero_si128() ;
	EB_U64 result;
	xmm1=_mm256_sad_epu8( xmm0 ,_mm256_set_m128i( _mm_loadl_epi64((__m128i *)(inputSamples+inputStride)) , _mm_loadl_epi64((__m128i *)(inputSamples)) ));
	xmm2= _mm256_sad_epu8(xmm0,_mm256_set_m128i(_mm_loadl_epi64((__m128i *)(inputSamples+3*inputStride)) ,_mm_loadl_epi64((__m128i *)(inputSamples+2*inputStride)) ) ) ;
	sum1 = _mm256_add_epi16(xmm1, xmm2);
	
	inputSamples += 4 * inputStride;
	
	xmm1= _mm256_sad_epu8(xmm0,_mm256_set_m128i( _mm_loadl_epi64((__m128i *)(inputSamples+inputStride)) , _mm_loadl_epi64((__m128i *)(inputSamples)) )) ;
	xmm2= _mm256_sad_epu8(xmm0, _mm256_set_m128i(_mm_loadl_epi64((__m128i *)(inputSamples+3*inputStride)) ,_mm_loadl_epi64((__m128i *)(inputSamples+2*inputStride)) ) );
	sum2 = _mm256_add_epi16(xmm1, xmm2);
	
    sum = _mm256_add_epi16(sum1, sum2);
	upper = _mm256_extractf128_si256(sum,1) ; //extract upper 128 bit
	upper = _mm_add_epi32(upper, _mm_srli_si128(upper, 8)); // shift 2nd 16 bits to the 1st and sum both
	
	lower = _mm256_extractf128_si256(sum,0) ; //extract lower 128 bit
	lower = _mm_add_epi32(lower, _mm_srli_si128(lower, 8)); // shift 2nd 16 bits to the 1st and sum both
	
	mean = _mm_add_epi32(lower,upper);
	
	(void)inputAreaWidth;
    (void)inputAreaHeight;
    
    result = (EB_U64)_mm_cvtsi128_si32(mean) << 2;
    return result;
	
} 

/********************************************************************************************************************************/
    void  ComputeIntermVarFour8x8_AVX2_INTRIN(
        EB_U8 *  inputSamples,
        EB_U16   inputStride,
        EB_U64 * meanOf8x8Blocks,      // mean of four  8x8
        EB_U64 * meanOfSquared8x8Blocks)  // meanSquared
    {

        __m256i ymm1, ymm2, ymm3, ymm4, ymm_sum1, ymm_sum2, ymm_FinalSum,ymm_shift,/* ymm_blockMeanSquared*///,
                ymm_in,ymm_in_2S,ymm_in_second,ymm_in_2S_second,ymm_shiftSquared,ymm_permute8,
                ymm_result,ymm_blockMeanSquaredlow,ymm_blockMeanSquaredHi,ymm_inputlo,ymm_inputhi;
        
        __m128i ymm_blockMeanSquaredlo,ymm_blockMeanSquaredhi,ymm_resultlo,ymm_resulthi;
       
        __m256i ymm_zero = _mm256_setzero_si256();
        __m128i xmm_zero = _mm_setzero_si128();

        ymm_in    = _mm256_loadu_si256((__m256i *) inputSamples);
        ymm_in_2S = _mm256_loadu_si256((__m256i *)(inputSamples + 2 * inputStride));
        
        ymm1 = _mm256_sad_epu8(ymm_in, ymm_zero);
        ymm2 = _mm256_sad_epu8(ymm_in_2S, ymm_zero);

        ymm_sum1 = _mm256_add_epi16(ymm1, ymm2);

        inputSamples += 4 * inputStride;
        ymm_in_second    = _mm256_loadu_si256((__m256i *)inputSamples);
        ymm_in_2S_second = _mm256_loadu_si256((__m256i *)(inputSamples + 2* inputStride));

        ymm3 = _mm256_sad_epu8(ymm_in_second, ymm_zero);
        ymm4 = _mm256_sad_epu8(ymm_in_2S_second, ymm_zero);

        ymm_sum2 = _mm256_add_epi16(ymm3, ymm4);

        ymm_FinalSum = _mm256_add_epi16(ymm_sum1, ymm_sum2);

        ymm_shift = _mm256_set_epi64x (3,3,3,3 );
        ymm_FinalSum = _mm256_sllv_epi64(ymm_FinalSum,ymm_shift);

        _mm256_storeu_si256((__m256i *)(meanOf8x8Blocks), ymm_FinalSum);

        /*******************************Squared Mean******************************/
        
        ymm_inputlo = _mm256_unpacklo_epi8(ymm_in, ymm_zero);
        ymm_inputhi = _mm256_unpackhi_epi8(ymm_in, ymm_zero);
	    
        ymm_blockMeanSquaredlow = _mm256_madd_epi16(ymm_inputlo, ymm_inputlo);
        ymm_blockMeanSquaredHi  = _mm256_madd_epi16(ymm_inputhi, ymm_inputhi);

        ymm_inputlo = _mm256_unpacklo_epi8(ymm_in_2S, ymm_zero);
	    ymm_inputhi = _mm256_unpackhi_epi8(ymm_in_2S, ymm_zero);
        
        ymm_blockMeanSquaredlow = _mm256_add_epi32(ymm_blockMeanSquaredlow, _mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = _mm256_add_epi32(ymm_blockMeanSquaredHi, _mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_inputlo = _mm256_unpacklo_epi8(ymm_in_second, ymm_zero);
	    ymm_inputhi = _mm256_unpackhi_epi8(ymm_in_second, ymm_zero);
        
        ymm_blockMeanSquaredlow = _mm256_add_epi32(ymm_blockMeanSquaredlow, _mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = _mm256_add_epi32(ymm_blockMeanSquaredHi, _mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_inputlo = _mm256_unpacklo_epi8(ymm_in_2S_second, ymm_zero);
        ymm_inputhi = _mm256_unpackhi_epi8(ymm_in_2S_second, ymm_zero);
	    
        ymm_blockMeanSquaredlow = _mm256_add_epi32(ymm_blockMeanSquaredlow, _mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = _mm256_add_epi32(ymm_blockMeanSquaredHi, _mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_blockMeanSquaredlow = _mm256_add_epi32(ymm_blockMeanSquaredlow, _mm256_srli_si256(ymm_blockMeanSquaredlow, 8));
	    ymm_blockMeanSquaredHi  = _mm256_add_epi32(ymm_blockMeanSquaredHi, _mm256_srli_si256(ymm_blockMeanSquaredHi, 8));

        ymm_blockMeanSquaredlow = _mm256_add_epi32(ymm_blockMeanSquaredlow, _mm256_srli_si256(ymm_blockMeanSquaredlow, 4));
        ymm_blockMeanSquaredHi  = _mm256_add_epi32(ymm_blockMeanSquaredHi, _mm256_srli_si256(ymm_blockMeanSquaredHi, 4));

        ymm_permute8            = _mm256_set_epi32(0,0,0,0,0,0,4,0);
        ymm_blockMeanSquaredlow =  _mm256_permutevar8x32_epi32(ymm_blockMeanSquaredlow,ymm_permute8/*8*/);
        ymm_blockMeanSquaredHi  =  _mm256_permutevar8x32_epi32(ymm_blockMeanSquaredHi,ymm_permute8);
        
        ymm_blockMeanSquaredlo = _mm256_extracti128_si256(ymm_blockMeanSquaredlow,0); //lower 128
        ymm_blockMeanSquaredhi = _mm256_extracti128_si256(ymm_blockMeanSquaredHi,0); //lower 128

        ymm_result   = _mm256_unpacklo_epi32(_mm256_castsi128_si256(ymm_blockMeanSquaredlo),_mm256_castsi128_si256(ymm_blockMeanSquaredhi));
        ymm_resultlo = _mm_unpacklo_epi64(_mm256_castsi256_si128(ymm_result),xmm_zero);
        ymm_resulthi = _mm_unpackhi_epi64(_mm256_castsi256_si128(ymm_result),xmm_zero);
        
        
        ymm_result   = _mm256_set_m128i(ymm_resulthi,ymm_resultlo);
        
        ymm_permute8 = _mm256_set_epi32(7,5,6,4,3,1,2,0);
        ymm_result   =  _mm256_permutevar8x32_epi32(ymm_result,ymm_permute8);
        
        ymm_shiftSquared = _mm256_set1_epi64x (11 );

        ymm_result = _mm256_sllv_epi64(ymm_result,ymm_shiftSquared);
        

        _mm256_storeu_si256((__m256i *)(meanOfSquared8x8Blocks), ymm_result);


     }
