#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nchunks = get_num_groups(0) / ne10;
    int g       = get_group_id(0);
    int i10     = g / nchunks;
    int chunk   = g - i10 * nchunks;
    int i11     = get_group_id(1);
    int i12     = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global float * dst_row = (global float *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
    global float * src_row = (global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);

    int span  = (ne00 + nchunks - 1) / nchunks;
    int start = chunk * span;
    int end   = min(start + span, ne00);

    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int NL = 2;

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        if (ind >= ne00) {
            return;
        }
        dequantize_q4_0_f32(
            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03)) + ind/NL, ind%NL, &temp);
        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

//------------------------------------------------------------------------------
// get_rows on the flattened (SoA) Q1_0 / Q2_0 weight: GluRun (patch 0006).
// src0_q holds the 16 quant bytes of every block back to back, src0_d one half
// per block; a row's first block is its byte offset in the original tensor
// divided by the 18-byte block. One work-group per output row, one work-item
// per block.
//------------------------------------------------------------------------------
#define QK1_0 128
#define QK2_0 64

kernel void kernel_get_rows_q1_0_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    ulong ib0 = ((ulong)r*nb01 + (ulong)i02*nb02 + (ulong)i03*nb03) / (QK1_0/8 + 2);
    global float * out = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int ind = get_local_id(0); ind < ne00/QK1_0; ind += get_local_size(0)) {
        ulong ib = ib0 + ind;
        float d = (float) src0_d[ib];
        global uchar * q = src0_q + ib*(QK1_0/8);
        global float * o = out + ind*QK1_0;
        for (int i = 0; i < QK1_0/8; ++i) {
            uint b = q[i];
            for (int j = 0; j < 8; ++j) {
                o[8*i + j] = ((b >> j) & 1) ? d : -d;
            }
        }
    }
}

kernel void kernel_get_rows_q2_0_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    ulong ib0 = ((ulong)r*nb01 + (ulong)i02*nb02 + (ulong)i03*nb03) / (QK2_0/4 + 2);
    global float * out = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int ind = get_local_id(0); ind < ne00/QK2_0; ind += get_local_size(0)) {
        ulong ib = ib0 + ind;
        float d = (float) src0_d[ib];
        global uchar * q = src0_q + ib*(QK2_0/4);
        global float * o = out + ind*QK2_0;
        for (int i = 0; i < QK2_0/4; ++i) {
            uint b = q[i];
            o[4*i + 0] = d * ((float)((b >> 0) & 3) - 1.0f);
            o[4*i + 1] = d * ((float)((b >> 2) & 3) - 1.0f);
            o[4*i + 2] = d * ((float)((b >> 4) & 3) - 1.0f);
            o[4*i + 3] = d * ((float)((b >> 6) & 3) - 1.0f);
        }
    }
}

//------------------------------------------------------------------------------
// get_rows on the Adreno-transposed Q1_0 / Q2_0 weight (GluRun, patch 0006):
// uint u of row r (32 / 16 codes) at u*ne01 + r, the scale of block kb at
// kb*ne01 + r (set_tensor's transpose_2d_as_32b / _16b). Strided reads, fine
// for an embedding lookup. 2-D weights only (the transpose needs ne2 = ne3 = 1).
//------------------------------------------------------------------------------
kernel void kernel_get_rows_q1_0_trans(
        global uint  * src0_q,
        global half  * src0_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
    global float * out = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int kb = get_local_id(0); kb < ne00/QK1_0; kb += get_local_size(0)) {
        float d = (float) src0_d[(ulong)kb*ne01 + r];
        global float * o = out + kb*QK1_0;
        for (int u = 0; u < 4; ++u) {
            uint w = src0_q[((ulong)4*kb + u)*ne01 + r];
            for (int j = 0; j < 32; ++j) {
                o[32*u + j] = ((w >> j) & 1) ? d : -d;
            }
        }
    }
}

kernel void kernel_get_rows_q2_0_trans(
        global uint  * src0_q,
        global half  * src0_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
    global float * out = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int kb = get_local_id(0); kb < ne00/QK2_0; kb += get_local_size(0)) {
        float d = (float) src0_d[(ulong)kb*ne01 + r];
        global float * o = out + kb*QK2_0;
        for (int u = 0; u < 4; ++u) {
            uint w = src0_q[((ulong)4*kb + u)*ne01 + r];
            for (int j = 0; j < 16; ++j) {
                o[16*u + j] = d * ((float)((w >> (2*j)) & 3) - 1.0f);
            }
        }
    }
}
