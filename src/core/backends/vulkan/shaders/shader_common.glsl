#ifndef SHADER_COMMON_GLSL
#define SHADER_COMMON_GLSL

// Required extensions
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require

// Precision settings
#ifdef USE_FP16
#extension GL_EXT_shader_16bit_storage : require
#define PRECISION float16_t
#else
#define PRECISION float
#endif

// Optimization hints
#define UNROLL_HINT [[unroll]]
#define LOOP_HINT(X) [[loop(X)]]
#define FLATTEN_HINT [[flatten]]

// Shared memory optimization
#define SHARED_MEM_BANK_SIZE 32
#define AVOID_BANK_CONFLICT(idx) ((idx) + ((idx) >> 5))

// Memory access patterns
struct MemoryBarrier {
    void barrier {
        memoryBarrierShared();
        groupMemoryBarrier();
        barrier();
    }
    
    void compute {
        memoryBarrierShared();
        groupMemoryBarrier();
    }
};

// Efficient reduction operations
PRECISION reduce_sum(PRECISION x) {
    uint lane_id = gl_SubgroupInvocationID;
    uint active_mask = gl_SubgroupEqMask;
    
    UNROLL_HINT
    for (uint offset = gl_SubgroupSize/2; offset > 0; offset >>= 1) {
        x += subgroupShuffleDown(x, offset);
    }
    
    return subgroupBroadcastFirst(x);
}

// Efficient matrix operations
void matrix_multiply_tile(
    in PRECISION[16] A,
    in PRECISION[16] B,
    inout PRECISION[16] C,
    uint tile_size
) {
    UNROLL_HINT
    for (uint i = 0; i < tile_size; i++) {
        UNROLL_HINT
        for (uint j = 0; j < tile_size; j++) {
            PRECISION sum = 0;
            UNROLL_HINT
            for (uint k = 0; k < tile_size; k++) {
                sum += A[i * tile_size + k] * B[k * tile_size + j];
            }
            C[i * tile_size + j] = sum;
        }
    }
}

// Efficient tensor operations
struct TensorOps {
    static PRECISION dot_product(
        in PRECISION[] a,
        in PRECISION[] b,
        uint size
    ) {
        PRECISION sum = 0;
        UNROLL_HINT
        for (uint i = 0; i < size; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    static void scale_add(
        inout PRECISION[] dst,
        in PRECISION[] src,
        PRECISION scale,
        uint size
    ) {
        UNROLL_HINT
        for (uint i = 0; i < size; i++) {
            dst[i] += src[i] * scale;
        }
    }
};

// Efficient interpolation
PRECISION interpolate_linear(
    PRECISION a,
    PRECISION b,
    PRECISION t
) {
    return a + (b - a) * t;
}

// Cache-friendly memory access
uint get_strided_index(uint idx, uint stride) {
    return (idx % stride) * (gl_WorkGroupSize.x / stride) + (idx / stride);
}

// Advanced optimization settings
#define WORKGROUP_SIZE 256
#define TILE_SIZE 16
#define VECTOR_WIDTH 4

// Vectorized types for better memory access
#ifdef USE_FP16
#define VEC_TYPE vec4_float16_t
#else
#define VEC_TYPE vec4
#endif

// Advanced memory barriers
struct MemoryBarrierEx {
    void global() {
        memoryBarrier();
        barrier();
    }
    
    void compute_to_compute() {
        memoryBarrier();
        groupMemoryBarrier();
        barrier();
    }
    
    void shared_only() {
        memoryBarrierShared();
        groupMemoryBarrier();
    }
    
    void buffer_only() {
        memoryBarrierBuffer();
        groupMemoryBarrier();
    }
};

// Vectorized operations
VEC_TYPE vector_load(in PRECISION[] data, uint index) {
    return VEC_TYPE(
        data[index],
        data[index + 1],
        data[index + 2],
        data[index + 3]
    );
}

void vector_store(inout PRECISION[] data, uint index, VEC_TYPE value) {
    data[index] = value.x;
    data[index + 1] = value.y;
    data[index + 2] = value.z;
    data[index + 3] = value.w;
}

// Optimized matrix operations with tiling
void matrix_multiply_tiled(
    in PRECISION[] A,
    in PRECISION[] B,
    inout PRECISION[] C,
    uint M, uint N, uint K
) {
    // Shared memory tiles
    shared PRECISION A_tile[TILE_SIZE][TILE_SIZE];
    shared PRECISION B_tile[TILE_SIZE][TILE_SIZE];
    
    uint row = gl_LocalInvocationID.x;
    uint col = gl_LocalInvocationID.y;
    uint global_row = gl_GlobalInvocationID.x;
    uint global_col = gl_GlobalInvocationID.y;
    
    PRECISION sum = 0.0;
    
    // Iterate over tiles
    for (uint t = 0; t < K; t += TILE_SIZE) {
        // Load tile cooperatively
        if (global_row < M && (t + col) < K)
            A_tile[row][col] = A[global_row * K + t + col];
        if ((t + row) < K && global_col < N)
            B_tile[row][col] = B[(t + row) * N + global_col];
            
        barrier();
        
        // Compute tile
        UNROLL_HINT
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[row][k] * B_tile[k][col];
        }
        
        barrier();
    }
    
    // Store result
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

// Advanced tensor operations
struct TensorOpsEx {
    // Vectorized dot product
    static PRECISION dot_product_vec(
        in PRECISION[] a,
        in PRECISION[] b,
        uint size
    ) {
        PRECISION sum = 0;
        uint vec_size = size / VECTOR_WIDTH;
        
        UNROLL_HINT
        for (uint i = 0; i < vec_size; i++) {
            VEC_TYPE va = vector_load(a, i * VECTOR_WIDTH);
            VEC_TYPE vb = vector_load(b, i * VECTOR_WIDTH);
            sum += dot(va, vb);
        }
        
        // Handle remaining elements
        for (uint i = vec_size * VECTOR_WIDTH; i < size; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    // Fused multiply-add with vectorization
    static void fused_multiply_add(
        inout PRECISION[] dst,
        in PRECISION[] src1,
        in PRECISION[] src2,
        PRECISION scale,
        uint size
    ) {
        uint vec_size = size / VECTOR_WIDTH;
        
        UNROLL_HINT
        for (uint i = 0; i < vec_size; i++) {
            uint idx = i * VECTOR_WIDTH;
            VEC_TYPE v1 = vector_load(src1, idx);
            VEC_TYPE v2 = vector_load(src2, idx);
            vector_store(dst, idx, fma(v1, v2, VEC_TYPE(scale)));
        }
        
        // Handle remaining elements
        for (uint i = vec_size * VECTOR_WIDTH; i < size; i++) {
            dst[i] = fma(src1[i], src2[i], scale);
        }
    }
};

// Advanced interpolation with SIMD
VEC_TYPE interpolate_linear_vec(
    VEC_TYPE a,
    VEC_TYPE b,
    PRECISION t
) {
    return mix(a, b, VEC_TYPE(t));
}

// Cache-optimized strided access
uint get_optimized_index(uint idx, uint stride, uint group_size) {
    uint block = idx / group_size;
    uint offset = idx % group_size;
    return (block * stride + offset) * VECTOR_WIDTH;
}

// Efficient reduction with subgroups
PRECISION reduce_sum_subgroup(PRECISION x) {
    uint sg_size = gl_SubgroupSize;
    
    UNROLL_HINT
    for (uint offset = sg_size/2; offset > 0; offset >>= 1) {
        x += subgroupShuffleDown(x, offset);
    }
    
    if (gl_SubgroupInvocationID == 0) {
        shared PRECISION partial_sums[MAX_COMPUTE_WORK_GROUP_SIZE];
        partial_sums[gl_WorkGroupID.x] = x;
    }
    
    barrier();
    
    if (gl_LocalInvocationID.x < gl_NumSubgroups) {
        x = partial_sums[gl_LocalInvocationID.x];
        
        UNROLL_HINT
        for (uint offset = gl_NumSubgroups/2; offset > 0; offset >>= 1) {
            x += subgroupShuffleDown(x, offset);
        }
    }
    
    return subgroupBroadcastFirst(x);
}

#endif // SHADER_COMMON_GLSL
