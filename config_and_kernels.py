"""
Configuration constants and CUDA kernel code for Continuous Evolution.
"""

# --- CONFIGURATION ---
MAX_PLAYERS = 1000
STARTING_PLAYERS = 200
MAX_ENEMIES = 6
SWORD_COUNT = 15
MAX_SWORDS = MAX_PLAYERS * SWORD_COUNT

TOTAL_ENTITIES = MAX_PLAYERS + 1 + MAX_ENEMIES + MAX_SWORDS
SCREEN_W, SCREEN_H = 800, 600

# Entity Indexes
IDX_P_START = 0
IDX_SHOP = MAX_PLAYERS
IDX_E_START = MAX_PLAYERS + 1
IDX_S_START = MAX_PLAYERS + 1 + MAX_ENEMIES

# Neural Network Config
INPUTS = 18
OUTPUTS = 7

# Gameplay Network (MLP) Config
N_IN = 18
N_HID = 64
N_OUT = 7
TOTAL_PARAMS = (N_IN * N_HID) + N_HID + (N_HID * N_OUT) + N_OUT
SIGMA = 0.05

# Planning Networks Config
STAT_PLAN_HID = 32
STAT_PLAN_OUT = 10
STAT_PLAN_PARAMS = (1 * STAT_PLAN_HID) + STAT_PLAN_HID + (STAT_PLAN_HID * STAT_PLAN_OUT) + STAT_PLAN_OUT

SHOP_PLAN_HID = 32
SHOP_PLAN_OUT = 4
SHOP_PLAN_PARAMS = (1 * SHOP_PLAN_HID) + SHOP_PLAN_HID + (SHOP_PLAN_HID * SHOP_PLAN_OUT) + SHOP_PLAN_OUT

# --- CUDA KERNELS ---
CUDA_SRC = r'''
#define PI 3.14159265f
#define SCREEN_W 800.0f
#define SCREEN_H 600.0f
#define INPUT_COUNT 18
#define OUTPUT_COUNT 7

// --- HELPER FUNCTIONS ---
__device__ unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ float atomicAdd(float* address, float val) {
    unsigned int* addr_as_ull = (unsigned int*)address;
    unsigned int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}
#endif

// --- NEURAL NETWORK INFERENCE KERNELS ---
extern "C" __global__ void dense_inference(float* inputs, float* all_weights, float* outputs, int max_players) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= max_players) return;

    int w_offset = idx * ${TOTAL_PARAMS};
    float hidden[${N_HID}];
    float in_local[${N_IN}];

    for(int i=0; i<${N_IN}; i++) in_local[i] = inputs[(idx * ${N_IN}) + i];

    for(int h=0; h<${N_HID}; h++) {
        float sum = 0.0f;
        for(int i=0; i<${N_IN}; i++) sum += in_local[i] * all_weights[w_offset + (i * ${N_HID} + h)];
        sum += all_weights[w_offset + (${N_IN} * ${N_HID}) + h];
        hidden[h] = fast_tanh(sum);
    }
    w_offset += (${N_IN} * ${N_HID}) + ${N_HID};

    for(int o=0; o<${N_OUT}; o++) {
        float sum = 0.0f;
        for(int h=0; h<${N_HID}; h++) sum += hidden[h] * all_weights[w_offset + (h * ${N_OUT} + o)];
        sum += all_weights[w_offset + (${N_HID} * ${N_OUT}) + o];
        outputs[(idx * ${N_OUT}) + o] = (o < 3) ? fast_tanh(sum) : sum;
    }
}

extern "C" __global__ void stat_plan_inference(float* all_weights, float* outputs, int max_players) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= max_players) return;

    int w_offset = idx * ${STAT_PLAN_PARAMS};
    float hidden[${STAT_PLAN_HID}];
    float input = 1.0f; 

    for(int h=0; h<${STAT_PLAN_HID}; h++) {
        float sum = input * all_weights[w_offset + h];
        sum += all_weights[w_offset + ${STAT_PLAN_HID} + h]; 
        hidden[h] = fast_tanh(sum);
    }
    w_offset += ${STAT_PLAN_HID} + ${STAT_PLAN_HID};

    for(int o=0; o<${STAT_PLAN_OUT}; o++) {
        float sum = 0.0f;
        for(int h=0; h<${STAT_PLAN_HID}; h++) {
            sum += hidden[h] * all_weights[w_offset + (h * ${STAT_PLAN_OUT} + o)];
        }
        sum += all_weights[w_offset + (${STAT_PLAN_HID} * ${STAT_PLAN_OUT}) + o];
        float val = fast_tanh(sum); 
        val = (val + 1.0f) * 1.5f; 
        outputs[(idx * ${STAT_PLAN_OUT}) + o] = val;
    }
}

extern "C" __global__ void shop_plan_inference(float* all_weights, float* outputs, int max_players) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= max_players) return;

    int w_offset = idx * ${SHOP_PLAN_PARAMS};
    float hidden[${SHOP_PLAN_HID}];
    float input = 1.0f;

    for(int h=0; h<${SHOP_PLAN_HID}; h++) {
        float sum = input * all_weights[w_offset + h];
        sum += all_weights[w_offset + ${SHOP_PLAN_HID} + h]; 
        hidden[h] = fast_tanh(sum);
    }
    w_offset += ${SHOP_PLAN_HID} + ${SHOP_PLAN_HID};

    for(int o=0; o<${SHOP_PLAN_OUT}; o++) {
        float sum = 0.0f;
        for(int h=0; h<${SHOP_PLAN_HID}; h++) {
            sum += hidden[h] * all_weights[w_offset + (h * ${SHOP_PLAN_OUT} + o)];
        }
        sum += all_weights[w_offset + (${SHOP_PLAN_HID} * ${SHOP_PLAN_OUT}) + o];
        outputs[(idx * ${SHOP_PLAN_OUT}) + o] = sum;
    }
}

extern "C" __global__ void pack_render_single(
    float* pos_x, float* pos_y, float* angle, float* active, float* type, 
    float* out_buffer, int total_entities
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_entities) return;

    int b = gid * 6;
    out_buffer[b+0] = pos_x[gid];
    out_buffer[b+1] = pos_y[gid];
    out_buffer[b+2] = angle[gid];

    float t = type[gid];
    float size = 20.0f; 
    if (t < 0.5f) size = 25.0f; // Player
    else if (t < 1.5f) size = 20.0f; // Enemy
    else if (t < 2.5f) size = 15.0f; // Sword
    else size = 45.0f; // Shop

    out_buffer[b+3] = size;
    out_buffer[b+4] = t;
    out_buffer[b+5] = active[gid];
}
'''