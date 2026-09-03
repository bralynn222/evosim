import sys
import os

# Force Optimus/high-performance GPU BEFORE SDL creates the GL context.
# Windows hybrid laptops default to Intel iGPU; without this the GL renderer
# is Intel and CUDA/GL interop fails -> CPU fallback + vsync 30fps.
# SHIM_MCCOMPAT is the actual switch that worked on RTX 2070 Max-Q
# (verified: without -> Intel, with -> NVIDIA).
os.environ.setdefault("SHIM_MCCOMPAT", "0x800000001")
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GL_DEVICE_OFFSET", "0")

# Set CUDA environment variables for CuPy. The toolkit paths below are the
# author's Linux install; they are meaningless on Windows (where cupy-cuda12x
# bundles the CUDA runtime), so only apply them when the directory exists.
_cuda_toolkit = '/home/bralynn/miniconda3/envs/myenv'
if os.path.isdir(_cuda_toolkit):
    if 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = _cuda_toolkit
    if 'CUPY_CUDA_INC_PATH' not in os.environ:
        os.environ['CUPY_CUDA_INC_PATH'] = _cuda_toolkit + '/include'
    _cu_lib = _cuda_toolkit + '/lib'
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = _cu_lib
    elif _cu_lib not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':' + _cu_lib

import ctypes
import warnings
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders

# cupy-cuda12x bundles the CUDA runtime; the "CUDA path could not be detected"
# UserWarning is cosmetic on machines without a standalone CUDA toolkit install.
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

try:
    import cupy as cp
except ImportError:
    print("CRITICAL: CuPy not found. This simulation requires an NVIDIA GPU and CuPy.")
    sys.exit(1)

from cupy.cuda.memory import MemoryPointer, UnownedMemory

from config_and_kernels import *
from physics_kernels import PHYSICS_KERNELS
from evolution_core import evolution_pass, log_deaths, render_kernel_source

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


def _load_cuda_gl_interop():
    """Bind CUDA Graphics/OpenGL interop functions via ctypes. None if unavailable.

    Uses the CUDA *runtime* API (cudart) rather than the driver API (nvcuda).
    On some Turing laptop GPUs (RTX 20-series Max-Q) with recent drivers,
    cuGraphicsResourceGetMappedPointer (driver API) fails with
    CUDA_ERROR_INVALID_CONTEXT (201) even though the resource maps fine, but
    the equivalent cudaGraphicsResourceGetMappedPointer (runtime API) works.
    """
    # Prefer the cudart bundled with cupy so the same runtime instance (and
    # thus the current CUDA context) is shared with all cupy operations.
    if os.name == "nt":
        names = ["cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll"]
        bundled_dll = "cudart64_*.dll"
    else:
        names = ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        bundled_dll = "libcudart.so*"

    candidates = list(names)
    # Locate cupy's bundled CUDA runtime (pip meta-package nvidia-cuda-runtime).
    try:
        import glob as _glob
        import nvidia.cuda_runtime as _crt
        for _p in getattr(_crt, "__path__", []):
            candidates += _glob.glob(os.path.join(_p, "**", bundled_dll), recursive=True)
    except Exception:
        pass
    if os.name == "nt" and not candidates:
        try:
            import site as _site
            for _p in _site.getsitepackages() + [_site.getusersitepackages()]:
                candidates += _glob.glob(os.path.join(_p, "**", "cudart64_*.dll"), recursive=True)
        except Exception:
            pass

    lib = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None

    def bind(name, restype, argtypes):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            return None
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    class CU:
        pass

    cu = CU()
    cu.register_buffer = bind("cudaGraphicsGLRegisterBuffer", ctypes.c_int,
                              [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint])
    cu.map = bind("cudaGraphicsMapResources", ctypes.c_int,
                  [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    cu.get_ptr = bind("cudaGraphicsResourceGetMappedPointer", ctypes.c_int,
                      [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p])
    cu.unmap = bind("cudaGraphicsUnmapResources", ctypes.c_int,
                    [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    cu.unregister = bind("cudaGraphicsUnregisterResource", ctypes.c_int, [ctypes.c_void_p])
    if any(x is None for x in (cu.register_buffer, cu.map, cu.get_ptr, cu.unmap)):
        return None
    return cu


def _force_nvidia_gpu():
    """Reroute OpenGL onto the NVIDIA adapter on hybrid (Optimus) laptops.

    pygame/SDL creates its GL context on the default display adapter, which on
    hybrid laptops is the Intel iGPU. That context is not CUDA-capable, so
    cuGraphicsGLRegisterBuffer fails and we silently fall back to CPU uploads.
    Force the current python.exe onto the high-performance (NVIDIA) GPU via the
    Windows per-app graphics preference registry, and set SHIM_MCCOMPAT /
    __GL_DEVICE_OFFSET. Requires one relaunch the first time so the OS applies
    the saved preference.
    """
    if sys.platform != "win32":
        return
    try:
        import winreg

        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        # Win10 2004+ ignores basename-only entries; must use full path.
        # Write both for compatibility.
        exes = [sys.executable, os.path.basename(sys.executable).lower()]
        # dedupe preserving order
        seen = set()
        exes = [x for x in exes if not (x in seen or seen.add(x))]
        want = "GpuPreference=2;"

        changed = False
        for exe in exes:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                                    winreg.KEY_READ | winreg.KEY_SET_VALUE) as k:
                    try:
                        cur = winreg.QueryValueEx(k, exe)[0]
                    except FileNotFoundError:
                        cur = None
                    if cur != want:
                        winreg.SetValueEx(k, exe, 0, winreg.REG_SZ, want)
                        changed = True
            except FileNotFoundError:
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as k:
                    winreg.SetValueEx(k, exe, 0, winreg.REG_SZ, want)
                changed = True

        os.environ["SHIM_MCCOMPAT"] = "0x800000001"
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__GL_DEVICE_OFFSET"] = "0"

        if changed and not os.environ.get("EVOSIM_GPU_FIXED"):
            os.environ["EVOSIM_GPU_FIXED"] = "1"
            print("GPU fix: set high-performance GPU preference for this app; "
                  "relaunching once so it takes effect...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        pass


class Simulation:
    def __init__(self):
        # Compile CUDA kernels (template rendering performed in Cython)
        self.mod = cp.RawModule(code=render_kernel_source(
            CUDA_SRC, TOTAL_PARAMS, N_IN, N_HID, N_OUT,
            STAT_PLAN_PARAMS, STAT_PLAN_HID, STAT_PLAN_OUT,
            SHOP_PLAN_PARAMS, SHOP_PLAN_HID, SHOP_PLAN_OUT) + PHYSICS_KERNELS)

        self.k_brain = self.mod.get_function('dense_inference')
        self.k_stat_plan = self.mod.get_function('stat_plan_inference')
        self.k_shop_plan = self.mod.get_function('shop_plan_inference')
        self.k_compact = self.mod.get_function('compact_active')
        self.k_upd_players = self.mod.get_function('update_players')
        self.k_upd_swords = self.mod.get_function('update_swords')
        self.k_spawn_swords = self.mod.get_function('spawn_player_swords')
        self.k_upd_enemies = self.mod.get_function('update_enemies')
        self.k_pack = self.mod.get_function('pack_render_single')

        # GPU Memory Allocations
        self.pos_x = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.pos_y = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.angle = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.active = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.type = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.timers = cp.zeros(TOTAL_ENTITIES, cp.float32)
        self.shop_timer = cp.zeros(MAX_PLAYERS, cp.float32)

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

        # Compacted active-entity lists (built each frame, drive bounded launches)
        self.active_players = cp.zeros(MAX_PLAYERS, dtype=cp.int32)
        self.active_swords = cp.zeros(MAX_SWORDS, dtype=cp.int32)
        self.player_count = cp.zeros(1, dtype=cp.int32)
        self.sword_count = cp.zeros(1, dtype=cp.int32)

        # CPU Trackers for thresholds
        self.death_check_count = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self.clone_check_count = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self.frame = 0

        # Cython evolution-pass scratch buffers (host arrays)
        self._ev_killed = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self._ev_parents = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self._ev_children = np.zeros(MAX_PLAYERS, dtype=np.int32)
        self._ev_children_dev = cp.zeros(MAX_PLAYERS, dtype=cp.int32)

        self.all_idx = cp.arange(MAX_PLAYERS, dtype=cp.int32)

        # UI caption stats refreshed by the (30-frame) evolution pass, host side
        self.stats = {'active_count': 0, 'best_kills': 0, 'best_age_frames': 0}

        self.init_world()

    def init_world(self):
        """Setup initial entities and seed population (all GPU-side)."""
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

        # Seed initial players entirely on the GPU
        n = STARTING_PLAYERS
        self.pos_x[:n] = cp.random.uniform(100, 700, n, dtype=cp.float32)
        self.pos_y[:n] = cp.random.uniform(100, 500, n, dtype=cp.float32)
        self.age[:n] = 0
        self.interval_kills[:n] = 0
        self.total_kills[:n] = 0
        self.exp[:n] = 0
        self.level[:n] = 1
        self.stat_points[:n] = 0
        self.stat_speed[:n] = 0
        self.stat_atk[:n] = 0
        self.stat_hp_max[:n] = 1
        self.current_hp[:n] = 1
        self.gold[:n] = 0
        self.triple_shot_flag[:n] = 0
        self.invincibility[:n] = 0
        self.timers[:n] = 0
        self.shop_timer[:n] = 0
        self.active[:n] = 1
        self.death_check_count[:n] = 0
        self.clone_check_count[:n] = 0

        self.generate_all_plans(self.all_idx, MAX_PLAYERS)

    def generate_all_plans(self, player_list, count):
        """(Re)compute stat + shop plans only for the given player slots.

        player_list: device int32 array of player indices; count: entries used.
        Full-range for initial seeding, affected-children-only on clone events.
        """
        grid = (count + 255) // 256

        self.k_stat_plan((grid,), (256,), (self.stat_plan_weights.ravel(), self.stat_plan_outputs,
                                           player_list, count))
        rows = player_list[:count].astype(cp.int64)
        gi = rows[:, None] * 10 + cp.arange(10, dtype=cp.int64)
        gi = gi.ravel()
        self.stat_plan_choices[gi] = cp.clip(self.stat_plan_outputs[gi], 0, 3).astype(cp.int64)

        self.k_shop_plan((grid,), (256,), (self.shop_plan_weights.ravel(), self.shop_plan_outputs,
                                           player_list, count))
        gi4 = (rows[:, None] * 4 + cp.arange(4, dtype=cp.int64)).ravel()
        vals = self.shop_plan_outputs[gi4].reshape(count, 4)
        self.shop_plan_priority[gi4] = cp.argsort(-vals, axis=1).astype(cp.int64).ravel()

    def continuous_evolution(self):
        """CPU side handling of births and deaths. Loops run in Cython.

        One batched device->host drain per 30 frames: the same host copies
        that feed the evolution pass also refresh the UI caption stats, so no
        extra GPU reductions are needed.
        """
        active_h = self.active[:MAX_PLAYERS].get()
        age_h = self.age.get()
        kills_h = self.interval_kills.get()
        tkills_h = self.total_kills.get()

        # Reset tracking stats of every inactive slot so that if a clone
        # activates such a slot later in this same pass, the loop cannot read
        # stale age/kills/counters from its previous life and kill it instantly.
        dead = active_h < 0.5
        age_h[dead] = 0.0
        kills_h[dead] = 0.0
        self.death_check_count[dead] = 0
        self.clone_check_count[dead] = 0

        n_killed, n_clones, extinct = evolution_pass(
            active_h, age_h, kills_h,
            self.death_check_count, self.clone_check_count,
            self._ev_killed,
            self._ev_parents, self._ev_children,
            MAX_PLAYERS,
        )

        # Push back active state (deaths + clone activations) and interval
        # kills (30s-window resets) in bulk.
        self.active[:MAX_PLAYERS] = cp.asarray(active_h)
        self.interval_kills[:] = cp.asarray(kills_h)

        log_deaths(self._ev_killed, n_killed, kills_h)

        # Clone and Mutate (fully on the GPU, no Python loop, no host sync)
        if n_clones > 0:
            c = self._ev_children[:n_clones]
            p = self._ev_parents[:n_clones]

            self.gameplay_weights[c] = (self.gameplay_weights[p] +
                                        cp.random.randn(n_clones, TOTAL_PARAMS, dtype=cp.float32) * SIGMA)
            self.stat_plan_weights[c] = (self.stat_plan_weights[p] +
                                         cp.random.randn(n_clones, STAT_PLAN_PARAMS, dtype=cp.float32) * SIGMA)
            self.shop_plan_weights[c] = (self.shop_plan_weights[p] +
                                         cp.random.randn(n_clones, SHOP_PLAN_PARAMS, dtype=cp.float32) * SIGMA)

            # Respawn at parent position + jitter, reset state -- all device writes
            self.pos_x[c] = self.pos_x[p] + cp.random.uniform(-30, 30, n_clones, dtype=cp.float32)
            self.pos_y[c] = self.pos_y[p] + cp.random.uniform(-30, 30, n_clones, dtype=cp.float32)
            self.age[c] = 0
            self.interval_kills[c] = 0
            self.total_kills[c] = 0
            self.exp[c] = 0
            self.level[c] = 1
            self.stat_points[c] = 0
            self.stat_speed[c] = 0
            self.stat_atk[c] = 0
            self.stat_hp_max[c] = 1
            self.current_hp[c] = 1
            self.gold[c] = 0
            self.triple_shot_flag[c] = 0
            self.invincibility[c] = 0
            self.timers[c] = 0
            self.shop_timer[c] = 0
            self.death_check_count[c] = 0
            self.clone_check_count[c] = 0

            # Recompute plans only for the newly cloned children (their weights
            # just changed); adults' plans stay cached.
            self._ev_children_dev[:n_clones] = cp.asarray(c)
            self.generate_all_plans(self._ev_children_dev, n_clones)

            tkills_h[c] = 0
            age_h[c] = 0

        # Host-side split stats for the UI caption (no device reductions)
        alive = np.flatnonzero(active_h > 0.5)
        if alive.size:
            self.stats = {
                'active_count': int(alive.size),
                'best_kills': int(tkills_h[alive].max()),
                'best_age_frames': int(age_h[alive].max()),
            }
        else:
            self.stats = {'active_count': 0, 'best_kills': 0, 'best_age_frames': 0}

        # Anti-Extinction
        if extinct:
            print(f"Extinction at frame {self.frame}! Respawning population...")
            self.init_world()

    def update(self):
        self.frame += 1

        # 0. Compact active entities so all later kernels only touch live data
        self.player_count.fill(0)
        self.sword_count.fill(0)
        grid_p = (MAX_PLAYERS + 255) // 256
        grid_s = (MAX_SWORDS + 255) // 256
        self.k_compact((grid_p,), (256,), (self.active, self.active_players, self.player_count, 0, MAX_PLAYERS))

        # 1. GPU Brain Inference (active players only)
        self.k_brain((grid_p,), (256,), (self.inputs, self.gameplay_weights.ravel(), self.outputs,
                                         self.active_players, self.player_count, MAX_PLAYERS))

        # 2. Physics & Logic
        self.k_upd_players((grid_p,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers, self.shop_timer,
            self.age, self.interval_kills, self.total_kills,
            self.exp, self.level, self.stat_points, self.stat_speed, self.stat_atk, self.stat_hp_max,
            self.current_hp, self.invincibility, self.gold, self.triple_shot_flag,
            self.inputs, self.outputs, self.p_rng, self.stat_plan_choices, self.shop_plan_priority,
            self.active_players, self.player_count,
            MAX_PLAYERS, MAX_ENEMIES
        ))

        grid_swords = (MAX_SWORDS + 255) // 256
        self.k_spawn_swords((grid_p,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.outputs, self.stat_atk, self.triple_shot_flag, self.p_rng,
            self.active_players, self.player_count,
            MAX_PLAYERS, MAX_ENEMIES, SWORD_COUNT
        ))

        self.sword_count.fill(0)
        self.k_compact((grid_s,), (256,), (self.active, self.active_swords, self.sword_count, IDX_S_START, MAX_SWORDS))
        self.k_upd_swords((grid_swords,), (256,), (
            self.pos_x, self.pos_y, self.angle, self.active, self.timers,
            self.active_swords, self.sword_count
        ))

        grid_enemies = (MAX_ENEMIES + 255) // 256
        self.k_upd_enemies((grid_enemies,), (256,), (
            self.pos_x, self.pos_y, self.active,
            self.current_hp, self.invincibility,
            self.interval_kills, self.total_kills, self.exp, self.gold,
            self.rng,
            self.active_players, self.player_count, self.active_swords, self.sword_count,
            MAX_ENEMIES, MAX_PLAYERS, MAX_SWORDS, SWORD_COUNT
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
        _force_nvidia_gpu()  # must happen before the GL context is created
        pygame.init()
        # Disable vsync: Intel 30fps was half-vsync from >16ms frame on iGPU.
        # Must be set before set_mode, and also forced after via SDL.
        try:
            pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
        except Exception:
            pass
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.DOUBLEBUF | pygame.OPENGL)
        # pygame's gl_set_attribute is ignored on some SDL2/pygame-ce builds;
        # force it off via SDL2 directly (verified: flips go 30fps -> 2700fps).
        try:
            SDL = ctypes.CDLL("SDL2.dll")
            SDL.SDL_GL_SetSwapInterval.argtypes = [ctypes.c_int]
            SDL.SDL_GL_SetSwapInterval.restype = ctypes.c_int
            SDL.SDL_GL_SetSwapInterval(0)
        except Exception:
            try:
                ctypes.windll.opengl32.wglSwapIntervalEXT(0)
            except Exception:
                pass
        self.clock = pygame.time.Clock()
        self._gl_renderer = glGetString(GL_RENDERER)
        if isinstance(self._gl_renderer, bytes):
            self._gl_renderer = self._gl_renderer.decode("utf-8", "replace")
        print(f"OpenGL renderer: {self._gl_renderer}")

        self.sim = Simulation()

        self.prog = shaders.compileProgram(
            shaders.compileShader(VERT, GL_VERTEX_SHADER),
            shaders.compileShader(FRAG, GL_FRAGMENT_SHADER)
        )

        q = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, q.nbytes, q, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Two instance VBOs (+ a VAO each) for double buffering the render
        # upload: CUDA writes buffer A while OpenGL draws buffer B, so the
        # GPU never has to fully drain before a frame can be presented.
        _zero_instance = np.zeros(TOTAL_ENTITIES * 6, dtype=np.float32)
        self.vaos = []
        self.ivbos = []
        for _ in range(2):
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
            inst = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, inst)
            glBufferData(GL_ARRAY_BUFFER, _zero_instance.nbytes, _zero_instance, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
            glVertexAttribDivisor(1, 1)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(16))
            glVertexAttribDivisor(2, 1)
            self.vaos.append(vao)
            self.ivbos.append(inst)
        glBindVertexArray(0)

        # Try to register both VBOs with CUDA so pack_render_single writes the
        # VBO directly (zero-copy). Falls back to CPU upload when the GL context
        # is not backed by the CUDA device (e.g. Intel iGPU + discrete NVIDIA).
        # Flag 2 = CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD (whole buffer is
        # rewritten by the pack kernel every frame).
        self._render_buf_cpu = self.sim.render_buf
        self._cu = _load_cuda_gl_interop()
        if self._cu is not None:
            self._gl_resources = []
            cu_err = 0
            for ivbo in self.ivbos:
                res = ctypes.c_void_p()
                cu_err = self._cu.register_buffer(ctypes.byref(res), ivbo, 2)
                if cu_err == 0:
                    cu_err = self._cu.map(1, ctypes.byref(res), 0)
                    if cu_err == 0:
                        ptr = ctypes.c_void_p()
                        size = ctypes.c_size_t()
                        cu_err = self._cu.get_ptr(ctypes.byref(ptr), ctypes.byref(size), res)
                        self._cu.unmap(1, ctypes.byref(res), 0)
                        if cu_err == 0 and size.value != TOTAL_ENTITIES * 24:
                            cu_err = -1
                if cu_err != 0:
                    self._cu = None
                    break
                self._gl_resources.append(res)
            if self._cu is None or len(self._gl_resources) != 2:
                self._cu = None
                print("CUDA/OpenGL interop unavailable (GL needs a CUDA-backed "
                      f"context); using CPU render upload. (cudaGraphics err={cu_err}, "
                      f"GL renderer={self._gl_renderer!r})")
            else:
                self._events = [cp.cuda.Event(disable_timing=True) for _ in range(2)]

    def draw_ui(self):
        """Update window caption from host stats cached by the evolution pass."""
        fps = self.clock.get_fps()
        s = self.sim.stats
        caption = (f"Continuous ES | Players: {s['active_count']}/{MAX_PLAYERS} | "
                   f"Best Kills: {s['best_kills']} | "
                   f"Best Age: {s['best_age_frames'] // 60}s | FPS: {fps:.0f}")
        pygame.display.set_caption(caption)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

            w = (self.sim.frame + 1) % 2  # instance VBO written this frame

            updated = False
            if self._cu is not None:
                # Let CUDA write whole frame into VBO[w] (pack renders into the
                # wrapped memory), then hand it back to GL without a wait.
                cu = self._cu
                if cu.map(1, ctypes.byref(self._gl_resources[w]), 0) != 0:
                    self._cu = None
                else:
                    try:
                        ptr = ctypes.c_void_p()
                        size = ctypes.c_size_t()
                        if cu.get_ptr(ctypes.byref(ptr), ctypes.byref(size),
                                      self._gl_resources[w]) != 0:
                            raise RuntimeError("cuGraphicsResourceGetMappedPointer failed")
                        mem = MemoryPointer(UnownedMemory(ptr.value, size.value, self), 0)
                        self.sim.render_buf = cp.ndarray((TOTAL_ENTITIES * 6,), dtype=cp.float32, memptr=mem)
                        self.sim.update()
                        updated = True
                        self._events[w].record(cp.cuda.get_current_stream())
                    except Exception:
                        self._cu = None
                    finally:
                        cu.unmap(1, ctypes.byref(self._gl_resources[w]), 0)
            if not updated:
                self.sim.render_buf = self._render_buf_cpu
                self.sim.update()

            draw_buf = (w + 1) % 2  # the other VBO = the previous frame's write
            if self._cu is not None and self.sim.frame > 1:
                # Wait only on LAST frame's work (already done), never on the
                # work just enqueued this frame.
                self._events[draw_buf].synchronize()

            glClearColor(0.05, 0.05, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.prog)
            glUniform2f(glGetUniformLocation(self.prog, "res"), SCREEN_W, SCREEN_H)

            glBindVertexArray(self.vaos[draw_buf])
            if self._cu is None:
                data = self.sim.render_buf.get()
                glBindBuffer(GL_ARRAY_BUFFER, self.ivbos[draw_buf])
                glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, TOTAL_ENTITIES)

            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(0)


if __name__ == "__main__":
    App().run()