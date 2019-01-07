/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbPictureOperators_AVX2
#define EbPictureOperators_AVX2

#include "EbDefinitions.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void EB_ENC_msbPack2D_AVX2_INTRIN_AL(
	EB_U8     *in8BitBuffer,
	EB_U32     in8Stride,
	EB_U8     *innBitBuffer,
	EB_U16    *out16BitBuffer,
	EB_U32     innStride,
	EB_U32     outStride,
	EB_U32     width,
	EB_U32     height);


extern void CompressedPackmsb_AVX2_INTRIN(
	EB_U8     *in8BitBuffer,
	EB_U32     in8Stride,
	EB_U8     *innBitBuffer,
	EB_U16    *out16BitBuffer,
	EB_U32     innStride,
	EB_U32     outStride,
	EB_U32     width,
	EB_U32     height);


void CPack_AVX2_INTRIN(
	const EB_U8     *innBitBuffer,
	EB_U32     innStride,
	EB_U8     *inCompnBitBuffer,
	EB_U32     outStride,
	EB_U8    *localCache,
	EB_U32     width,
	EB_U32     height);


void UnpackAvg_AVX2_INTRIN(
	    EB_U16 *ref16L0,
        EB_U32  refL0Stride,
        EB_U16 *ref16L1,
        EB_U32  refL1Stride,
        EB_U8  *dstPtr,
        EB_U32  dstStride,      
        EB_U32  width,
        EB_U32  height);

EB_S32  sumResidual8bit_AVX2_INTRIN(
                     EB_S16 * inPtr,
                     EB_U32   size,
                     EB_U32   strideIn );
void memset16bitBlock_AVX2_INTRIN (
                    EB_S16 * inPtr,
                    EB_U32   strideIn,
                    EB_U32   size,
                    EB_S16   value
    );


void UnpackAvgSafeSub_AVX2_INTRIN(
	    EB_U16 *ref16L0,
        EB_U32  refL0Stride,
        EB_U16 *ref16L1,
        EB_U32  refL1Stride,
        EB_U8  *dstPtr,
        EB_U32  dstStride,
        EB_U32  width,
        EB_U32  height);



void FullDistortionKernel4x4_32bit_BT_AVX2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void FullDistortionKernel8x8_32bit_BT_AVX2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void FullDistortionKernel16MxN_32bit_BT_AVX2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void PictureAverageKernel_AVX2_INTRIN(
	EB_BYTE src0,
	EB_U32 src0Stride,
	EB_BYTE src1,
	EB_U32 src1Stride,
	EB_BYTE dst,
	EB_U32 dstStride,
	EB_U32 areaWidth,
	EB_U32 areaHeight);

void ResidualKernel4x4_AVX2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void ResidualKernel8x8_AVX2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void ResidualKernel16x16_AVX2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void ResidualKernel32x32_AVX2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight);

void ResidualKernel64x64_AVX2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight);

#ifdef __cplusplus
}
#endif
#endif // EbPictureOperators_AVX2
