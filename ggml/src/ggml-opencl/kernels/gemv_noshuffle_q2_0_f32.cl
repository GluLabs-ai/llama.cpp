#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// GluRun (patch 0006): Adreno GEMV for Q2_0 (ternary, QK2_0 = 64) on the
// transposed weight, the structure of gemv_noshuffle_q1_0_f32.cl. The quant
// buffer is transposed as 32-bit words (one uint = 16 weights of one row, word
// u of row r at u*M + r), the scales as 16-bit (one half per 64-block, block kb
// of row r at kb*M + r). Weight i of a word is ((w >> 2i) & 3) - 1.
#define QK2_0 64
#define N_SIMDGROUP 4

// 16 weights of one uint against activations regB (8 per fiber) of fibers lb, lb+1
#define dequantizeBlockAccum_q2(total, bits, scale, regB, lb)                                             \
    total += ((float)((bits >>  0) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+0); \
    total += ((float)((bits >>  2) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+0); \
    total += ((float)((bits >>  4) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+0); \
    total += ((float)((bits >>  6) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+0); \
    total += ((float)((bits >>  8) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+0); \
    total += ((float)((bits >> 10) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+0); \
    total += ((float)((bits >> 12) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+0); \
    total += ((float)((bits >> 14) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+0); \
    total += ((float)((bits >> 16) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+1); \
    total += ((float)((bits >> 18) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+1); \
    total += ((float)((bits >> 20) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+1); \
    total += ((float)((bits >> 22) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+1); \
    total += ((float)((bits >> 24) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+1); \
    total += ((float)((bits >> 26) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+1); \
    total += ((float)((bits >> 28) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+1); \
    total += ((float)((bits >> 30) & 3u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+1);


#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q2_0_f32(
        read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        read_only  image1d_buffer_t src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M;
    uint BLOCK_STRIDE_A = 4 * M;

    uint4  regA;
    half   regS;
    float8 regB;

    float totalSum = 0.0f;

    #pragma unroll 1
    for (uint kb = groupId; kb < (K / QK2_0); kb += N_SIMDGROUP) {
        regS = src0_d[gid + kb * LINE_STRIDE_A]; // each fiber loads its row's scale

        // first 8 fibers load 8 B values each -> 64 activations for this block
        if (slid < 8) {
            regB.s0123 = read_imagef(src1, (slid * 2 + kb * 16));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + kb * 16));
        }

        // load this row's 4 uint32 (64 x 2-bit codes)
        regA.s0 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;

        float scale = (float)regS;
        dequantizeBlockAccum_q2(totalSum, regA.s0, scale, regB, 0);
        dequantizeBlockAccum_q2(totalSum, regA.s1, scale, regB, 2);
        dequantizeBlockAccum_q2(totalSum, regA.s2, scale, regB, 4);
        dequantizeBlockAccum_q2(totalSum, regA.s3, scale, regB, 6);
    }

    // reduction in local memory, assumes #wave = N_SIMDGROUP = 4
    local float reduceLM[SIMDGROUP_WIDTH * 3];
    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // The x-grid is padded to CEIL_DIV(M,wavesize)*wavesize; guard the tail rows.
        if (gid < M) dst[gid] = totalSum;
    }
}
