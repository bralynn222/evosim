import sys
import os

# Set CUDA environment variables for CuPy
if 'CUDA_PATH' not in os.environ:
    os.environ['CUDA_PATH'] = '/home/bralynn/miniconda3/envs/myenv'
if 'CUPY_CUDA_INC_PATH' not in os.environ:
    os.environ['CUPY_CUDA_INC_PATH'] = '/home/bralynn/miniconda3/envs/myenv/include'
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '/home/bralynn/miniconda3/envs/myenv/lib'
else:
    if '/home/bralynn/miniconda3/envs/myenv/lib' not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':/home/bralynn/miniconda3/envs/myenv/lib'

import ctypes
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders

try:
    import cupy as cp
except ImportError:
    print("CRITICAL: CuPy not found. This simulation requires an NVIDIA GPU and CuPy.")
    sys.exit(1)

from config_and_kernels import *
from physics_kernels import PHYSICS_KERNELS

# --- SHADERS ---
VERT = """
#version 330 core
layout(location=0) in vec2 q;
layout(location=1) in vec4 p; 
layout(location=2) in vec2 t; 

uniform vec2 res;
out vec4 v_color;

void main() {
    if (t.y < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; } // Inactive

    float ang = p.z;
    float size = p.w;
    float type = t.x;

    mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
    vec2 local = q * size;
    vec2 world = p.xy + (rot * local);

    vec2 ndc = (world / res) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Color logic
    if (type < 0.5) v_color = vec4(0.2, 0.6, 1.0, 1.0); // Player = Blue
    else if (type < 1.5) v_color = vec4(1.0, 0.2, 0.2, 1.0); // Enemy = Red
    else if (type < 2.5) v_color = vec4(1.0, 1.0, 1.0, 1.0); // Sword = White
    else v_color = vec4(0.2, 1.0, 0.2, 1.0); // Shop = Green
}
"""

FRAG = """
#version 330 core
in vec4 v_color;
out vec4 color;
void main() { color = v_color; }
"""


class Simulation:
    def __init__(self):
        # REPLACE INSTEAD OF FORMAT TO AVOID CURLY BRACE KEYERROR
        f_src = CUDA_SRC
        replaces = {
            "${TOTAL_PARAMS}": str(TOTAL_PARAMS),
            "${N_IN}": str(N_IN),
            "${N_HID}": str(N_HID),
            "${N_OUT}": str(N_OUT),
            "${STAT_PLAN_PARAMS}": str(STAT_PLAN_PARAMS),
            "${STAT_PLAN_HID}": str(STAT_PLAN_HID),
            "${STAT_PLAN_OUT}": str(STAT_PLAN_OUT),
            "${SHOP_PLAN_PARAMS}": str(SHOP_PLAN_PARAMS),
            "${SHOP_PLAN_HID}": str(SHOP_PLAN_HID),
            "${SHOP_PLAN_OUT}": str(SHOP_PLAN_OUT)
        }
        for k, v in replaces.items():
            f_src = f_src.replace(k, v)

        # Compile CUDA kernels
        self.mod = cp.RawModule(code=f_src + PHYSICS_KERNELS)

        self.k_brain = self.mod.get_function('dense_inference')
        self.k_stat_plan = self.mod.get_function('stat_plan_inference')
        self.k_shop_plan = self.mod.get_function('shop_plan_inference')
        self.k_upd_players = self.mod.get_function('update_players')
        self.k_upd_swords = self.mod.get_function('update_swords')
        self.k_upd_enemies = self.mod.get_function('update_enemies')
        self.k_pack = self.mod.get_function('pack_render_single')

        # GPU Memory Allocations
        self.pos_x = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.pos_y = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.angle = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.active = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.type = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.timers = cp.zeros(TOTAL_ENTITIES, cp.float32)

        # Player specific stats
        self.age = cp.zeros(MAX_PLAYERS, cp.float32)
        self.interval_kills = cp.zeros(MAX_PLAYERS, cp.float32)
        self.total_kills = cp.zeros(MAX_PLAYERS, cp.float32)
        self.exp = cp.zeros(MAX_PLAYERS, cp.float32)
        self.level = cp.zeros(MAX_PLAYERS, cp.float32)
        self.stat_points = cp.zeros(MAX_PLAYERS, cp.float32)
        self.stat_speed = cp.zeros(MAX_PLAYERS, cp.float32)
        self.stat_atk = cp.zeros(MAX_PLAYERS, cp.float32)
        self.stat_hp_max = cp.zeros(MAX_PLAYERS, cp.float32)
        self.current_hp = cp.zeros(MAX_PLAYERS, cp.float32)
        self.invincibility = cp.zeros(MAX_PLAYERS, cp.float32)
        self.gold = cp.zeros(MAX_PLAYERS, cp.float32)
        self.triple_shot_flag = cp.zeros(MAX_PLAYERS, cp.float32)

        self.rng = cp.random.randint(0, 2 ** 30, MAX_ENEMIES, dtype=cp.uint32)
        self.p_rng = cp.random.randint(0, 2 ** 30, MAX_PLAYERS, dtype=cp.uint32)

        self.inputs = cp.zeros(MAX_PLAYERS * INPUTS, cp.float32)
        self.outputs = cp.zeros(MAX_PLAYERS * OUTPUTS, cp.float32)

        # Networks
        self.gameplay_weights = cp.random.randn(MAX_PLAYERS, TOTAL_PARAMS, dtype=cp.float32) * 0.1
        self.stat_plan_weights = cp.random.randn(MAX_PLAYERS, STAT_PLAN_PARAMS, dtype=cp.float32) * 0.1
        self.shop_plan_weights = cp.random.randn(MAX_PLAYERS, SHOP_PLAN_PARAMS, dtype=cp.float32) * 0.1

        self.stat_plan_outputs = cp.zeros(MAX_PLAYERS * 10, cp.float32)
        self.stat_plan_choices = cp.zeros(MAX_PLAYERS * 10, cp.int32)
        self.shop_plan_outputs = cp.zeros(MAX_PLAYERS * 4, cp.float32)
        self.shop_plan_priority = cp.zeros(MAX_PLAYERS * 4, cp.int32)

        self.render_buf = cp.zeros(TOTAL_ENTITIES * 6, dtype=cp.float32)

        # CPU Trackers for thresholds
        self.death_check_count = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self.clone_check_count = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self.frame = 0

        self.init_world()

    def init_world(self):
        """Setup initial entities and seed population."""
        self.active.fill(0)

        # Assign Types
        self.type[IDX_P_START:IDX_SHOP] = 0
        self.type[IDX_SHOP] = 3
        self.type[IDX_E_START:IDX_S_START] = 1
        self.type[IDX_S_START:] = 2

        # Initialize Shop
        self.active[IDX_SHOP] = 1.0
        self.pos_x[IDX_SHOP] = SCREEN_W / 2
        self.pos_y[IDX_SHOP] = SCREEN_H / 2

        # Seed initial players
        for i in range(STARTING_PLAYERS):
            self.reset_player(i)
            self.active[i] = 1.0

        self.generate_all_plans()

    def reset_player(self, idx, parent_x=None, parent_y=None):
        if parent_x is None:
            self.pos_x[idx] = float(np.random.uniform(100, 700))
            self.pos_y[idx] = float(np.random.uniform(100, 500))
        else:
            self.pos_x[idx] = float(parent_x + np.random.uniform(-30, 30))
            self.pos_y[idx] = float(parent_y + np.random.uniform(-30, 30))

        self.age[idx] = 0
        self.interval_kills[idx] = 0
        self.total_kills[idx] = 0
        self.exp[idx] = 0
        self.level[idx] = 1
        self.stat_points[idx] = 0
        self.stat_speed[idx] = 0
        self.stat_atk[idx] = 0
        self.stat_hp_max[idx] = 1
        self.current_hp[idx] = 1
        self.gold[idx] = 0
        self.triple_shot_flag[idx] = 0
        self.invincibility[idx] = 0

        self.death_check_count[idx] = 0
        self.clone_check_count[idx] = 0

    def generate_all_plans(self):
        grid = (MAX_PLAYERS + 255) // 256
        self.k_stat_plan((grid,), (256,), (self.stat_plan_weights.ravel(), self.stat_plan_outputs, MAX_PLAYERS))
        self.stat_plan_choices = cp.asarray(
            np.clip(self.stat_plan_outputs.get().reshape(MAX_PLAYERS, 10).astype(int), 0, 3).ravel())

        self.k_shop_plan((grid,), (256,), (self.shop_plan_weights.ravel(), self.shop_plan_outputs, MAX_PLAYERS))
        self.shop_plan_priority = cp.asarray(
            np.argsort(-self.shop_plan_outputs.get().reshape(MAX_PLAYERS, 4), axis=1).ravel())

    def continuous_evolution(self):
        """CPU Side handling of births and deaths."""
        active_h = self.active[:MAX_PLAYERS].get()
        age_h = self.age.get()
        kills_h = self.interval_kills.get()

        cloned_any = False

        for i in range(MAX_PLAYERS):
            if active_h[i] < 0.5: continue

            frames_alive = int(age_h[i])

            # Rule: Death if AI fails to get 3 new kills every 30 seconds (1800 frames)
            periods_30s = frames_alive // 1800
            if periods_30s > self.death_check_count[i]:
                self.death_check_count[i] = periods_30s
                if kills_h[i] < 3:
                    print(f"Player {i} killed by '30s Kill Rule' (Kills: {int(kills_h[i])}/3)")
                    active_h[i] = 0.0
                    self.active[i] = 0.0
                    continue
                else:
                    # Reset interval kills for the next 30s window
                    self.interval_kills[i] = 0.0

            # Rule: Clone if survived 15 seconds (900 frames)
            periods_15s = frames_alive // 900
            if periods_15s > self.clone_check_count[i]:
                self.clone_check_count[i] = periods_15s

                empty_slots = np.where(active_h < 0.5)[0]
                if len(empty_slots) > 0:
                    c = empty_slots[0]
                    # Clone and Mutate
                    self.gameplay_weights[c] = self.gameplay_weights[i] + cp.random.randn(TOTAL_PARAMS,
                                                                                          dtype=cp.float32) * SIGMA
                    self.stat_plan_weights[c] = self.stat_plan_weights[i] + cp.random.randn(STAT_PLAN_PARAMS,
                                                                                            dtype=cp.float32) * SIGMA
                    self.shop_plan_weights[c] = self.shop_plan_weights[i] + cp.random.randn(SHOP_PLAN_PARAMS,
                                                                                            dtype=cp.float32) * SIGMA

                    self.reset_player(c, parent_x=self.pos_x[i].item(), parent_y=self.pos_y[i].item())
                    active_h[c] = 1.0
                    self.active[c] = 1.0
                    cloned_any = True

        if cloned_any:
            self.generate_all_plans()

        # Anti-Extinction
        if np.sum(active_h) == 0:
            print(f"Extinction at frame {self.frame}! Respawning population...")
            self.init_world()

    def update(self):
        self.frame += 1

        # 1. GPU Brain Inference
        grid_pop = (MAX_PLAYERS + 255) // 256
        self.k_brain((grid_pop,), (256,), (self.inputs, self.gameplay_weights.ravel(), self.outputs, MAX_PLAYERS))

        # 2. Physics & Logic
        self.k_upd_players((grid_pop,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.age, self.interval_kills, self.total_kills,
            self.exp, self.level, self.stat_points, self.stat_speed, self.stat_atk, self.stat_hp_max,
            self.current_hp, self.invincibility, self.gold, self.triple_shot_flag,
            self.inputs, self.outputs, self.p_rng, self.stat_plan_choices, self.shop_plan_priority,
            MAX_PLAYERS, MAX_ENEMIES
        ))

        grid_swords = (MAX_SWORDS + 255) // 256
        self.k_upd_swords((grid_swords,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.outputs, self.stat_atk, self.triple_shot_flag, self.p_rng,
            MAX_SWORDS, MAX_PLAYERS, MAX_ENEMIES, SWORD_COUNT
        ))

        grid_enemies = (MAX_ENEMIES + 255) // 256
        self.k_upd_enemies((grid_enemies,), (256,), (
            self.pos_x, self.pos_y, self.active,
            self.current_hp, self.invincibility,
            self.interval_kills, self.total_kills, self.exp, self.gold,
            self.rng, MAX_ENEMIES, MAX_PLAYERS, MAX_SWORDS
        ))

        # 3. Handle Cloning and Death (Every 30 frames to save CPU-GPU sync time)
        if self.frame % 30 == 0:
            self.continuous_evolution()

        # 4. Pack for Render
        grid_render = (TOTAL_ENTITIES + 255) // 256
        self.k_pack((grid_render,), (256,),
                    (self.pos_x, self.pos_y, self.angle, self.active, self.type, self.render_buf, TOTAL_ENTITIES))


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.DOUBLEBUF | pygame.OPENGL)
        self.clock = pygame.time.Clock()

        self.sim = Simulation()

        self.prog = shaders.compileProgram(
            shaders.compileShader(VERT, GL_VERTEX_SHADER),
            shaders.compileShader(FRAG, GL_FRAGMENT_SHADER)
        )

        q = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, q.nbytes, q, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.ivbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ivbo)
        glBufferData(GL_ARRAY_BUFFER, TOTAL_ENTITIES * 24, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(16))
        glVertexAttribDivisor(2, 1)

    def draw_ui(self):
        """Update window caption with simulation stats."""
        active_h = self.sim.active[:MAX_PLAYERS].get()
        active_count = int(np.sum(active_h))

        if active_count > 0:
            total_kills_h = self.sim.total_kills.get()
            age_h = self.sim.age.get()
            # Only consider active players for stats
            mask = active_h > 0.5
            max_kills = int(np.max(total_kills_h[mask])) if np.any(mask) else 0
            max_age_frames = int(np.max(age_h[mask])) if np.any(mask) else 0
            max_age_s = max_age_frames // 60
        else:
            max_kills = 0
            max_age_s = 0

        fps = self.clock.get_fps()
        caption = f"Continuous ES | Players: {active_count}/{MAX_PLAYERS} | Best Kills: {max_kills} | Best Age: {max_age_s}s | FPS: {fps:.0f}"
        pygame.display.set_caption(caption)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

            self.sim.update()

            glClearColor(0.05, 0.05, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.prog)
            glUniform2f(glGetUniformLocation(self.prog, "res"), SCREEN_W, SCREEN_H)

            glBindVertexArray(self.vao)
            data = self.sim.render_buf.get()

            glBindBuffer(GL_ARRAY_BUFFER, self.ivbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, TOTAL_ENTITIES)

            if self.sim.frame % 30 == 0:
                self.draw_ui()

            pygame.display.flip()
            self.clock.tick(0)


if __name__ == "__main__":
    App().run()