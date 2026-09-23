#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// GluRun (patch 0006): Q2_0 (ternary, QK2_0 = 64) mat-vec on the flattened
// (SoA) weight: src0_q holds the 16 quant bytes of every block back to back,
// src0_d one half per block. Structure of mul_mv_q1_0_f32_flat.cl; the 2-bit
// code q of element j (byte j/4, bit 2*(j%4)) is worth (q - 1) * d.
#define QK2_0 64
#define QK2_0_BYTES (QK2_0/4)              // 16 quant bytes per block
#define QK2_0_BLK_BYTES (QK2_0_BYTES + 2)  // d + qs in original tensor = 18

#define NB_Q2_0 16 // quants handled per thread (one uint = four qs bytes)

#ifdef INTEL_GPU
#define N_R0_Q2_0 4 // number of rows each subgroup works on
#define N_SG_Q2_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q2_0 4
#define N_SG_Q2_0 2
#define N_SIMDWIDTH 64
#endif

// 16 quants from one uint (element i at bits 2i..2i+1), times the 16 activations
#define Q2_0_DOT16(acc, w, lo, hi)                                              \
    acc  = lo.s0*(float)((w >>  0) & 3u) + lo.s1*(float)((w >>  2) & 3u)       \
         + lo.s2*(float)((w >>  4) & 3u) + lo.s3*(float)((w >>  6) & 3u)       \
         + lo.s4*(float)((w >>  8) & 3u) + lo.s5*(float)((w >> 10) & 3u)       \
         + lo.s6*(float)((w >> 12) & 3u) + lo.s7*(float)((w >> 14) & 3u)       \
         + hi.s0*(float)((w >> 16) & 3u) + hi.s1*(float)((w >> 18) & 3u)       \
         + hi.s2*(float)((w >> 20) & 3u) + hi.s3*(float)((w >> 22) & 3u)       \
         + hi.s4*(float)((w >> 24) & 3u) + hi.s5*(float)((w >> 26) & 3u)       \
         + hi.s6*(float)((w >> 28) & 3u) + hi.s7*(float)((w >> 30) & 3u);

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q2_0_f32_flat(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK2_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q2_0 + get_sub_group_id()) * N_R0_Q2_0;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows (flat: q words + scales); block index = byte offset / 18
    ulong offset_src0_base = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global uint * ax0, * ax1, * ax2, * ax3;
    global half * ad0, * ad1, * ad2, * ad3;
    ulong offset_src0;

    offset_src0 = (offset_src0_base + 0*nb01) / QK2_0_BLK_BYTES;
    ax0 = (global uint *) ((global char *) src0_q + offset_src0*QK2_0_BYTES);
    ad0 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 1*nb01) / QK2_0_BLK_BYTES;
    ax1 = (global uint *) ((global char *) src0_q + offset_src0*QK2_0_BYTES);
    ad1 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 2*nb01) / QK2_0_BLK_BYTES;
    ax2 = (global uint *) ((global char *) src0_q + offset_src0*QK2_0_BYTES);
    ad2 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 3*nb01) / QK2_0_BLK_BYTES;
    ax3 = (global uint *) ((global char *) src0_q + offset_src0*QK2_0_BYTES);
    ad3 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    // 4 threads per block (16 quants = one uint each), N_SIMDWIDTH/4 blocks per step
    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    global float * yb = y + ix*QK2_0 + il*NB_Q2_0;

    float8 yl_lo;
    float8 yl_hi;
    float4 sumf = 0.f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        yl_lo = vload8(0, yb);
        yl_hi = vload8(0, yb + 8);
        float sumy = yl_lo.s0 + yl_lo.s1 + yl_lo.s2 + yl_lo.s3
                   + yl_lo.s4 + yl_lo.s5 + yl_lo.s6 + yl_lo.s7
                   + yl_hi.s0 + yl_hi.s1 + yl_hi.s2 + yl_hi.s3
                   + yl_hi.s4 + yl_hi.s5 + yl_hi.s6 + yl_hi.s7;

        uint w;
        float acc;

        w = ax0[ib*4 + il];
        Q2_0_DOT16(acc, w, yl_lo, yl_hi);
        sumf.s0 += (float)ad0[ib] * (acc - sumy);

        w = ax1[ib*4 + il];
        Q2_0_DOT16(acc, w, yl_lo, yl_hi);
        sumf.s1 += (float)ad1[ib] * (acc - sumy);

        w = ax2[ib*4 + il];
        Q2_0_DOT16(acc, w, yl_lo, yl_hi);
        sumf.s2 += (float)ad2[ib] * (acc - sumy);

        w = ax3[ib*4 + il];
        Q2_0_DOT16(acc, w, yl_lo, yl_hi);
        sumf.s3 += (float)ad3[ib] * (acc - sumy);

        yb += (N_SIMDWIDTH/4)*QK2_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = tot.s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = tot.s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = tot.s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = tot.s3;
    }
}
