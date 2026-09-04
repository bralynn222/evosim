```markdown
# G-EvoSim: High-Throughput Zero-Copy GPU Evolutionary Engine


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

## Tech Stack

* **Compute:** Custom CUDA C++ Kernels (RawModule compilation), CuPy
* **Low-Frequency CPU Loops:** Cython (C-extensions for bounds-checked genetic pass)
* **Graphics API:** Modern OpenGL (Core Profile 3.3), GLSL Instanced Arrays
* **Context & Platform Glue:** Pygame, ctypes (Direct binding to `nvcuda` / `cudart64_*.dll`)
* **Target Hardware:** NVIDIA GPUs (Turing, Ampere, Ada Lovelace, Hopper+)

---

## Project Structure


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
