```markdown
# G-EvoSim: High-Throughput Zero-Copy GPU Evolutionary Engine

[![Language](https://img.shields.io/badge/Language-CUDA%20%7C%20C%2B%2B%20%7C%20Cython%20%7C%20Python-blue.svg)](#)
[![Performance](https://img.shields.io/badge/Performance-2700%2B%20FPS-brightgreen.svg)](#)
[![Architecture](https://img.shields.io/badge/Compute-99.5%25%20GPU%20Resident-orange.svg)](#)

A massively parallel, hardware-accelerated evolutionary artificial intelligence simulation engine. Built to eliminate the host-device bottleneck entirely, **G-EvoSim** executes agent neural inference, physics, spatial queries, evolutionary genetic mutations, and rendering memory assembly directly in VRAM on the GPU.

By combining raw CUDA C++ kernels, stream compaction, low-overhead Cython orchestration, and **zero-copy CUDA/OpenGL interop via direct runtime pointer mapping**, simulation state never crosses the PCIe bus during normal execution loops.

---

## Key Highlights & Performance

* **>99.5% GPU Compute Residency:** Zero per-frame PCIe transfers (`cudaMemcpy` eliminated from the inner loop).
* **Zero-Copy Rendering Pipeline:** Physics and packing kernels write directly into an active OpenGL Vertex Buffer Object (VBO) mapped to CUDA virtual address space via `cudaGraphicsResourceGetMappedPointer`.
* **Warp-Aggregated Stream Compaction:** Entity filtering uses `__ballot_sync` and `__shfl_sync` shuffle intrinsics, reducing global memory atomic contention by up to 32x per warp.
* **Double-Buffered Asynchronous Dispatch:** CUDA compute streams and OpenGL draw calls execute concurrently across ping-pong VBOs synchronized via non-blocking `cudaEvent_t` barriers.
* **Hardware-Aware Driver Workarounds:** Programmatic DirectX User Preferences registry modification and hybrid graphics (NVIDIA Optimus) arbitration ensures runtime OpenGL context creation stays anchored to the high-performance discrete GPU.

---

## Architectural Overview

```
                      +---------------------------------------+
                      |         HOST (CPU / Cython)           |
                      |  - Low-Frequency (30-frame) Evolution |
                      |  - Window Event Loop & Dispatch       |
                      +-------------------+-------------------+
                                          | Non-blocking Enqueue
                                          v
+---------------------------------------------------------------------------------+
|                               DEVICE VRAM (CUDA)                                |
|                                                                                 |
|  +---------------------+        +--------------------+        +---------------+ |
|  | compact_active      | -----> | dense_inference    | -----> | update_       | |
|  | Warp-level shuffle  |        | Multi-Head MLP     |        | players/swords| |
|  +---------------------+        +--------------------+        +---------------+ |
|                                                                       |         |
|  +--------------------------------------------------------------------+         |
|  |                                                                              |
|  v                                                                              |
|  +---------------------+        +--------------------+        +---------------+ |
|  | update_enemies      | -----> | pack_render_single | -----> | Mapped OpenGL | |
|  | Spatial & HP checks |        | VBO Interop Kernel |        | VBO (Buffer A)| |
|  +---------------------+        +--------------------+        +---------------+ |
+-----------------------------------------------------------------------|---------+
                                                                        | Zero-Copy
                                                                        v Render
                                                                +---------------+
                                                                | OpenGL Context|
                                                                | Instanced VBO |
                                                                +---------------+
```

---

## Deep Dive: Low-Level Optimizations

### 1. Warp-Aggregated Stream Compaction
Simulating dynamic lifecycles (entities dying and spawning) typically causes severe warp divergence. Rather than scanning all entity slots or issuing thread-level `atomicAdd` instructions, the compaction pass uses intra-warp register communication:
```cpp
unsigned int amask = __ballot_sync(0xffffffffu, is_active);
int lane = threadIdx.x & 31;
int warp_count = __popc(amask);
int warp_offset = 0;
if (lane == 0 && warp_count > 0) {
    warp_offset = atomicAdd(out_count, warp_count); // 1 atomic per warp
}
warp_offset = __shfl_sync(0xffffffffu, warp_offset, 0);
if (is_active) {
    unsigned int prefix = amask & ((1u << lane) - 1);
    out_list[warp_offset + __popc(prefix)] = start + i;
}
```

### 2. Zero-Copy CUDA/OpenGL Interop
Traditional simulation loops read data back to host memory using `glBufferSubData` or NumPy arrays. G-EvoSim dynamically registers OpenGL instances with the CUDA driver runtime:
```python
# Mapped directly to GPU virtual memory
cu.register_buffer(ctypes.byref(res), ivbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)
cu.map(1, ctypes.byref(res), 0)
cu.get_ptr(ctypes.byref(ptr), ctypes.byref(size), res)

# Custom packed render kernel directly writes to VRAM buffer
self.sim.render_buf = cp.ndarray((TOTAL_ENTITIES * 6,), dtype=cp.float32, memptr=mem)
```

### 3. Transcendental Math Optimizations
All continuous actor activations replace expensive floating-point `tanh` instructions with a degree-5 Padé approximant (`fast_tanh`), compiling down to fast fused multiply-add (FMA) instructions:
$$\tanh(x) \approx \frac{x(135135 + x^2(17325 + x^2(378 + x^2)))}{135135 + x^2(62370 + x^2(3150 + 28x^2))}$$

---

## Tech Stack

* **Compute:** Custom CUDA C++ Kernels (RawModule compilation), CuPy
* **Low-Frequency CPU Loops:** Cython (C-extensions for bounds-checked genetic pass)
* **Graphics API:** Modern OpenGL (Core Profile 3.3), GLSL Instanced Arrays
* **Context & Platform Glue:** Pygame, ctypes (Direct binding to `nvcuda` / `cudart64_*.dll`)
* **Target Hardware:** NVIDIA GPUs (Turing, Ampere, Ada Lovelace, Hopper+)

---

## Project Structure

```
├── main.py                 # Engine bootstrap, CUDA-GL interop, loop orchestration
├── config_and_kernels.py   # Simulation configurations & inference CUDA source
├── physics_kernels.py      # Spatial partitioning, entity updates & compaction kernels
├── evolution_core.pyx      # Cython-accelerated host decision loops
├── setup.py                # C-Extension build script with platform-specific flags
└── README.md
```

---

## Getting Started

### Prerequisites
* NVIDIA GPU with current drivers
* CUDA Toolkit 11.x or 12.x
* Python 3.9+
* C++ Build Tools (MSVC for Windows / GCC for Linux)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/g-evosim.git
   cd g-evosim
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pygame pyopengl cupy-cuda12x cython setuptools
   ```
   *(Ensure your `cupy-cudaXX` version matches your local CUDA driver).*

3. **Build the Cython evolution module:**
   ```bash
   python setup.py build_ext --inplace
   ```

4. **Run the simulation:**
   ```bash
   python main.py
   ```

---

## Engineering Edge Cases Handled

* **NVIDIA Optimus Auto-Routing:** Hybrid laptops default to creating OpenGL contexts on the integrated Intel processor, causing interop failures. The application injects preferences directly into `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` to enforce discrete execution.
* **Driver SwapInterval Bypasses:** Circumvents driver-enforced 30/60 FPS VSync caps across problematic SDL2 dynamic linkings using direct native calls to `wglSwapIntervalEXT(0)` and `SDL_GL_SetSwapInterval(0)`.
```
