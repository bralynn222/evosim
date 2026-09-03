# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: language_level=3

"""
Cython-accelerated CPU logic for the Continuous Evolution simulation.

Owns the remaining host-side loops:
  * the death/clone decision pass (evolution_pass)
  * kill-rule logging (log_deaths)
  * CUDA source template rendering (render_kernel_source)

Everything else is GPU-only: weight mutation, spawn/reset writes, plan
generation, render packing, and UI stats all run as vectorized numpy/cupy
directly on the device, so no interpreted-Python player loops remain.
"""

from libc.stdio cimport printf, fflush, FILE


def evolution_pass(
        float[:] active_h,
        const float[:] age_h,
        float[:] kills_h,
        int[:] death_count,
        int[:] clone_count,
        int[:] killed_idx,
        int[:] clone_parents,
        int[:] clone_children,
        int max_players,
):
    """Run one death/clone decision pass over all players.

    Mutates active_h (deaths -> 0.0, clones -> 1.0) and kills_h (30s-window
    resets -> 0.0) in place; fills the compact killed/clone index buffers.
    Returns (n_killed, n_clones, extinct).
    """
    cdef int i, j, c
    cdef int n_killed = 0
    cdef int n_clones = 0
    cdef int extinct = 1
    cdef int frames_alive, periods_30s, periods_15s

    for i in range(max_players):
        if active_h[i] < 0.5:
            continue

        frames_alive = <int>age_h[i]

        # Rule: Death if AI fails to get 3 new kills every 30 seconds (1800 frames)
        periods_30s = frames_alive // 1800
        if periods_30s > death_count[i]:
            death_count[i] = periods_30s
            if kills_h[i] < 3.0:
                killed_idx[n_killed] = i
                n_killed += 1
                active_h[i] = 0.0
                continue
            else:
                # Reset interval kills for the next 30s window
                kills_h[i] = 0.0

        # Rule: Clone if survived 15 seconds (900 frames)
        periods_15s = frames_alive // 900
        if periods_15s > clone_count[i]:
            clone_count[i] = periods_15s

            # First inactive player slot (active_h is mutated during this pass)
            c = -1
            for j in range(max_players):
                if active_h[j] < 0.5:
                    c = j
                    break

            if c >= 0:
                clone_parents[n_clones] = i
                clone_children[n_clones] = c
                n_clones += 1
                active_h[c] = 1.0

    # Anti-Extinction: true only if no player is active after this pass
    for i in range(max_players):
        if active_h[i] >= 0.5:
            extinct = 0
            break

    return n_killed, n_clones, extinct


def log_deaths(const int[:] killed_idx, int n_killed, const float[:] kills_h):
    """Print the '30s Kill Rule' death messages (C printf, no Python loop)."""
    cdef int i, idx
    for i in range(n_killed):
        idx = killed_idx[i]
        printf("Player %d killed by '30s Kill Rule' (Kills: %d/3)\n",
               idx, <int>kills_h[idx])
    fflush(<FILE*>0)


def render_kernel_source(str f_src,
                         int total_params, int n_in, int n_hid, int n_out,
                         int stat_plan_params, int stat_plan_hid, int stat_plan_out,
                         int shop_plan_params, int shop_plan_hid, int shop_plan_out):
    """Render the CUDA template placeholders (own old main.py predicate loop)."""
    cdef object replaces = {
        "${TOTAL_PARAMS}": str(total_params),
        "${N_IN}": str(n_in),
        "${N_HID}": str(n_hid),
        "${N_OUT}": str(n_out),
        "${STAT_PLAN_PARAMS}": str(stat_plan_params),
        "${STAT_PLAN_HID}": str(stat_plan_hid),
        "${STAT_PLAN_OUT}": str(stat_plan_out),
        "${SHOP_PLAN_PARAMS}": str(shop_plan_params),
        "${SHOP_PLAN_HID}": str(shop_plan_hid),
        "${SHOP_PLAN_OUT}": str(shop_plan_out),
    }
    cdef object k, v
    for k, v in replaces.items():
        f_src = f_src.replace(k, v)
    return f_src
