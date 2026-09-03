"""
CUDA physics kernels for Continuous Multiplayer simulation.
"""

PHYSICS_KERNELS = r'''

// Keep this file self-contained: macros are provided by CUDA_SRC when
// concatenated, but compiled standalone this must still compile.
#ifndef PI
#define PI 3.14159265f
#endif
#ifndef SCREEN_W
#define SCREEN_W 800.0f
#endif
#ifndef SCREEN_H
#define SCREEN_H 600.0f
#endif
#ifndef INPUT_COUNT
#define INPUT_COUNT 18
#endif
#ifndef OUTPUT_COUNT
#define OUTPUT_COUNT 7
#endif

// 0. COMPACT ACTIVE ENTITIES (device-side indexed list + count)
//    Warp-aggregated: only lane 0 of each warp does a global atomicAdd, so
//    contention is one atomic per warp instead of one per active entity.
//    slot is guaranteed < total (== out_list capacity), so no OOB writes.
extern "C" __global__ void compact_active(
    float* active, int* out_list, int* out_count, int start, int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_active = (i < total) && (active[start + i] > 0.5f);

    unsigned int amask = __ballot_sync(0xffffffffu, is_active);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(amask);
    int warp_offset = 0;
    if (lane == 0 && warp_count > 0) {
        warp_offset = atomicAdd(out_count, warp_count);
    }
    warp_offset = __shfl_sync(0xffffffffu, warp_offset, 0);

    if (is_active) {
        unsigned int prefix = amask & ((1u << lane) - 1);
        out_list[warp_offset + __popc(prefix)] = start + i;
    }
}

// 1. UPDATE PLAYERS (Shared Single Arena)
extern "C" __global__ void update_players(
    float* pos_x, float* pos_y, float* angle, float* active, float* timers, float* shop_timer,
    float* age, float* interval_kills, float* total_kills,
    float* exp, float* level, float* stat_points, float* stat_speed, float* stat_atk, float* stat_hp_max,
    float* current_hp, float* invincibility, float* gold, float* triple_shot_flag,
    float* ai_inputs, float* ai_outputs, unsigned int* rng_state,
    int* stat_plan_choices, int* shop_plan_priority,
    int* active_players, int* player_count,
    int max_players, int max_enemies
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = player_count[0];
    if (k >= n) return;
    int p = active_players[k];
    if (active[p] < 0.5f) return;

    age[p] += 1.0f;

    // --- LEVEL UP LOGIC (while: resolve multi-level batches in one frame) ---
    while (exp[p] >= level[p] * 100.0f) {
        exp[p] -= level[p] * 100.0f;
        int lvl = (int)level[p];
        level[p] += 1.0f;

        if (lvl >= 1 && lvl <= 10) {
            int plan_idx = p * 10 + (lvl - 1);
            int choice = stat_plan_choices[plan_idx];
            
            if (choice == 0) stat_speed[p] += 1.0f;
            else if (choice == 1) stat_atk[p] += 1.0f;
            else if (choice == 2) { 
                stat_hp_max[p] += 1.0f; 
                current_hp[p] += 1.0f; 
            }
        } else {
            stat_points[p] += 1.0f;
        }
    }

    // Auto Spend all pending points (while: drain any multi-point backlog)
    while (stat_points[p] >= 1.0f) {
        int b = p * OUTPUT_COUNT;
        float v_spd = ai_outputs[b+4];
        float v_atk = ai_outputs[b+5];
        float v_hp  = ai_outputs[b+6];
        if (v_spd >= v_atk && v_spd >= v_hp) stat_speed[p] += 1.0f;
        else if (v_atk >= v_spd && v_atk >= v_hp) stat_atk[p] += 1.0f;
        else { stat_hp_max[p] += 1.0f; current_hp[p] += 1.0f; }
        stat_points[p] -= 1.0f;
    }

    // --- MOVEMENT ---
    float spd_mult = 1.0f + (stat_speed[p] * 0.1f);
    float move_speed = 4.0f * spd_mult;

    float mx = ai_outputs[p * OUTPUT_COUNT + 0];
    float my = ai_outputs[p * OUTPUT_COUNT + 1];
    float ma = ai_outputs[p * OUTPUT_COUNT + 2];

    pos_x[p] += mx * move_speed;
    pos_y[p] += my * move_speed;
    angle[p] = ma * PI;

    // Bounds
    if (pos_x[p] < 0) pos_x[p] = 0; if (pos_x[p] > SCREEN_W) pos_x[p] = SCREEN_W;
    if (pos_y[p] < 0) pos_y[p] = 0; if (pos_y[p] > SCREEN_H) pos_y[p] = SCREEN_H;

    timers[p] -= 1.0f;
    if (shop_timer[p] > 0.0f) shop_timer[p] -= 1.0f;
    if (invincibility[p] > 0.0f) invincibility[p] -= 1.0f;

    // --- SHOP LOGIC ---
    int shop_idx = max_players; 
    float sdx = pos_x[shop_idx] - pos_x[p];
    float sdy = pos_y[shop_idx] - pos_y[p];
    float shop_dist_sq = sdx*sdx + sdy*sdy;

    if (shop_dist_sq < 2500.0f && shop_timer[p] <= 0.0f) {
         bool bought = false;
         int priority_base = p * 4;
         for(int pr=0; pr<4; pr++) {
             int item = shop_plan_priority[priority_base + pr];
             if (item == 0 && gold[p] >= 50.0f) { gold[p] -= 50.0f; stat_speed[p] += 2.0f; bought = true; break; }
             else if (item == 1 && gold[p] >= 50.0f) { gold[p] -= 50.0f; stat_atk[p] += 1.0f; bought = true; break; }
             else if (item == 2 && gold[p] >= 50.0f) { gold[p] -= 50.0f; stat_hp_max[p] += 2.0f; current_hp[p] = stat_hp_max[p]; bought = true; break; }
             else if (item == 3 && gold[p] >= 2500.0f && triple_shot_flag[p] < 0.5f) { gold[p] -= 2500.0f; triple_shot_flag[p] = 1.0f; bought = true; break; }
         }
         if(bought) shop_timer[p] = 30.0f;
    }

    // --- SENSOR PACKING ---
    float c_dist[6], c_dx[6], c_dy[6];
    for(int k=0; k<6; k++) { c_dist[k] = 999999.0f; c_dx[k]=0.0f; c_dy[k]=0.0f; }

    int e_start = max_players + 1;
    for(int i=0; i<max_enemies; i++) {
        int e = e_start + i;
        if (active[e] > 0.5f) {
            float dx = pos_x[e] - pos_x[p];
            float dy = pos_y[e] - pos_y[p];
            float dsq = dx*dx + dy*dy;
            if (dsq < c_dist[5]) {
                c_dist[5] = dsq; c_dx[5] = dx; c_dy[5] = dy;
                for(int k=5; k>0; k--) {
                    if (c_dist[k] < c_dist[k-1]) {
                        float td=c_dist[k]; c_dist[k]=c_dist[k-1]; c_dist[k-1]=td;
                        float tx=c_dx[k]; c_dx[k]=c_dx[k-1]; c_dx[k-1]=tx;
                        float ty=c_dy[k]; c_dy[k]=c_dy[k-1]; c_dy[k-1]=ty;
                    }
                }
            }
        }
    }

    int b = p * INPUT_COUNT;
    ai_inputs[b+0] = pos_x[p] / 800.0f;
    ai_inputs[b+1] = pos_y[p] / 600.0f;
    ai_inputs[b+2] = current_hp[p] / max(1.0f, stat_hp_max[p]);
    ai_inputs[b+3] = (timers[p] <= 0.0f) ? 1.0f : -1.0f;
    for(int k=0; k<6; k++) {
        ai_inputs[b + 4 + (k*2)] = c_dx[k] / 800.0f;
        ai_inputs[b + 4 + (k*2) + 1] = c_dy[k] / 600.0f;
    }
    ai_inputs[b + 16] = sdx / 800.0f;
    ai_inputs[b + 17] = sdy / 600.0f;
}

// 2. UPDATE SWORDS -- two passes in the same stream (fly then spawn) so a
//    flying sword and its respawn can never race on the same slot.
// 2a. FLY: move/expire every live sword (owners may be dead).
extern "C" __global__ void update_swords(
    float* pos_x, float* pos_y, float* angle, float* active, float* timers,
    int* active_swords, int* sword_count
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = sword_count[0];
    if (k >= n) return;

    int gs = active_swords[k];
    timers[gs] -= 1.0f;
    if (timers[gs] <= 0.0f) active[gs] = 0.0f;
    pos_x[gs] += cosf(angle[gs]) * 15.0f;
    pos_y[gs] += sinf(angle[gs]) * 15.0f;
}

// 2b. SPAWN: fire logic only for active players; a sword slot is only reused
//     while it is free (matches the per-slot semantics of the original kernel).
extern "C" __global__ void spawn_player_swords(
    float* pos_x, float* pos_y, float* angle, float* active, float* timers,
    float* ai_outputs, float* stat_atk, float* triple_shot_flag, unsigned int* rng_state,
    int* active_players, int* player_count,
    int max_players, int max_enemies, int swords_per_player
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = player_count[0];
    if (k >= n) return;
    int p = active_players[k];
    if (active[p] < 0.5f) return;

    float cooldown = max(2.0f, 15.0f - stat_atk[p]);
    float trigger = ai_outputs[p * OUTPUT_COUNT + 3];

    if (trigger > 0.0f && timers[p] <= 0.0f) {
        unsigned int rng = rng_state[p];
        rng = xorshift32(&rng);

        int base = max_players + 1 + max_enemies + p * swords_per_player;
        bool has_triple = (triple_shot_flag[p] > 0.5f);
        bool fired = false;

        if (!has_triple) {
            int pick = rng % swords_per_player;
            if (active[base + pick] < 0.5f) {
                active[base + pick] = 1.0f;
                timers[base + pick] = 20.0f;
                pos_x[base + pick] = pos_x[p];
                pos_y[base + pick] = pos_y[p];
                angle[base + pick] = angle[p];
                fired = true;
            }
        } else {
            int group = rng % (swords_per_player / 3);
            for (int sub = 0; sub < 3; sub++) {
                int j = group * 3 + sub;
                if (active[base + j] < 0.5f) {
                    float off = (sub == 0) ? -0.2f : (sub == 2) ? 0.2f : 0.0f;
                    active[base + j] = 1.0f;
                    timers[base + j] = 20.0f;
                    pos_x[base + j] = pos_x[p];
                    pos_y[base + j] = pos_y[p];
                    angle[base + j] = angle[p] + off;
                    fired = true;
                }
            }
        }
        if (fired) timers[p] = cooldown;
        rng_state[p] = rng;
    }
}

// 3. UPDATE ENEMIES (scans only compacted live-player and live-sword lists)
extern "C" __global__ void update_enemies(
    float* pos_x, float* pos_y, float* active,
    float* current_hp, float* invincibility,
    float* interval_kills, float* total_kills, float* exp, float* gold,
    unsigned int* rng_state,
    int* active_players, int* player_count, int* active_swords, int* sword_count,
    int max_enemies, int max_players, int max_swords, int swords_per_player
) {
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (e_idx >= max_enemies) return;

    int global_e = max_players + 1 + e_idx;
    unsigned int rng = rng_state[e_idx];
    rng = xorshift32(&rng);

    if (active[global_e] < 0.5f) {
        if ((rng % 100) < 3) { // Constant spawn pressure
            active[global_e] = 1.0f;
            int side = rng % 4;
            if (side == 0) { pos_x[global_e] = rng % 800; pos_y[global_e] = -10; }
            else if (side == 1) { pos_x[global_e] = rng % 800; pos_y[global_e] = 610; }
            else if (side == 2) { pos_x[global_e] = -10; pos_y[global_e] = rng % 600; }
            else { pos_x[global_e] = 810; pos_y[global_e] = rng % 600; }
        }
        rng_state[e_idx] = rng;
        return;
    }

    // Find closest player over the compacted live list (order is not sorted;
    // only the minimum distance matters, so ordering is irrelevant here)
    int n = player_count[0];
    float min_dist = 999999.0f;
    int target_p = -1;
    for(int k=0; k<n; k++) {
        int p = active_players[k];
        float dx = pos_x[p] - pos_x[global_e];
        float dy = pos_y[p] - pos_y[global_e];
        float dsq = dx*dx + dy*dy;
        if(dsq < min_dist) { min_dist = dsq; target_p = p; }
    }

    if(target_p != -1) {
        float dx = pos_x[target_p] - pos_x[global_e];
        float dy = pos_y[target_p] - pos_y[global_e];
        float d = sqrtf(min_dist) + 0.0001f;
        pos_x[global_e] += (dx/d) * 2.0f;
        pos_y[global_e] += (dy/d) * 2.0f;

        // Player collision
        if (min_dist < 400.0f) {
            active[global_e] = 0.0f;
            unsigned int claimed = atomicCAS((unsigned int*)&invincibility[target_p], 0u,
                                             (unsigned int)__float_as_int(30.0f));
            if (claimed == 0u) {
                current_hp[target_p] -= 1.0f;
                if(current_hp[target_p] <= 0.0f) active[target_p] = 0.0f;
            }
        }
    }

    // Sword collision check (compacted live-sword list; when several swords hit
    // in the same frame the kill credit is non-deterministic -- acceptable)
    if(active[global_e] > 0.5f) {
        int sn = sword_count[0];
        for(int sk=0; sk<sn; sk++) {
            int global_s = active_swords[sk];
            if (active[global_s] < 0.5f) continue;
            float dx = pos_x[global_s] - pos_x[global_e];
            float dy = pos_y[global_s] - pos_y[global_e];
            if(dx*dx + dy*dy < 500.0f) {
                active[global_e] = 0.0f;
                int owner = (global_s - (max_players + 1 + max_enemies)) / swords_per_player;
                if (active[owner] > 0.5f) {
                    atomicAdd(&interval_kills[owner], 1.0f);
                    atomicAdd(&total_kills[owner], 1.0f);
                    atomicAdd(&exp[owner], 25.0f);
                    atomicAdd(&gold[owner], 10.0f);
                }
                break;
            }
        }
    }
    rng_state[e_idx] = rng;
}
'''