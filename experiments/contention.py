import multiprocessing
import time
import math
import sys
import numpy as np

# Try importing PyCUDA; fail gracefully with instructions if missing
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except Exception:
    PYCUDA_AVAILABLE = False

# ==========================================
# 1. CPU COMPUTE STRESSOR (Worker Function)
# ==========================================
def _cpu_worker(target_load):
    """Internal worker that runs on a single CPU core."""
    if target_load <= 0:
        return
    active_time = (target_load / 100.0) * 0.1
    sleep_time = (1.0 - target_load / 100.0) * 0.1
    
    while True:
        start_time = time.time()
        while time.time() - start_time < active_time:
            # Heavy mathematical loop to keep execution pipelines saturated
            _ = math.sqrt(98765.43) * math.sin(0.7)
        if sleep_time > 0:
            time.sleep(sleep_time)

def stress_cpu(intensity_percentage):
    """
    Spawns CPU stressors across ALL available logical threads.
    intensity_percentage: Int/Float from 0 to 100
    """
    if intensity_percentage <= 0:
        return []
    
    cores = multiprocessing.cpu_count()
    processes = []
    print(f"[+] Starting CPU stressor: {intensity_percentage}% load across {cores} threads.")
    
    for _ in range(cores):
        p = multiprocessing.Process(target=_cpu_worker, args=(intensity_percentage,))
        p.daemon = True
        p.start()
        processes.append(p)
    return processes

# ==========================================
# 2. SYSTEM RAM BANDWIDTH STRESSOR
# ==========================================
def _ram_worker(intensity_percentage):
    """Internal worker causing continuous L3 cache misses to flood memory channels."""
    base_elements = int(40000000 * (intensity_percentage / 100.0))
    if base_elements < 1000: base_elements = 1000
    
    while True:
        # Generate two huge distinct arrays in system memory
        arr1 = np.ones(base_elements, dtype=np.float32)
        arr2 = np.ones(base_elements, dtype=np.float32)
        # Array manipulation forces sequential Read/Write cycles across channels
        arr3 = arr1 + arr2 
        
        if intensity_percentage < 95:
            time.sleep(0.002)

def stress_ram_bandwidth(intensity_percentage):
    """
    Spawns RAM bandwidth stressors. Spawns 2 processes to parallelize memory bus saturation.
    intensity_percentage: Int/Float from 0 to 100
    """
    if intensity_percentage <= 0:
        return []
        
    print(f"[+] Starting RAM Bandwidth stressor: {intensity_percentage}% intensity.")
    processes = []
    # Spawning 2 memory workers is generally optimal to fully saturate dual/quad-channel memory layouts
    for _ in range(2):
        p = multiprocessing.Process(target=_ram_worker, args=(intensity_percentage,))
        p.daemon = True
        p.start()
        processes.append(p)
    return processes

# ==========================================
# 3. GPU COMPUTE CORE STRESSOR
# ==========================================
def stress_gpu_compute(intensity_percentage):
    """
    Launches a dedicated process trapping GPU execution cores in an arithmetic loop.
    Data stays strictly inside GPU L2 cache to keep VRAM idle.
    """
    if intensity_percentage <= 0:
        return []
    if not PYCUDA_AVAILABLE:
        print("[!] GPU Stress requested but PyCUDA/CUDA is not available.")
        return []

    def _gpu_comp_worker(intensity):
        # Re-initialize CUDA inside the child process context
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        size = 1024 
        host_data = np.ones(size, dtype=np.float32)
        gpu_data = cuda.mem_alloc(host_data.nbytes)
        cuda.memcpy_htod(gpu_data, host_data)
        
        loop_limit = int(60000 * (intensity / 100.0))
        if loop_limit < 1: loop_limit = 1
        
        mod = SourceModule(f"""
            __global__ void pure_compute_kernel(float *data, int loops) {{
                int idx = threadIdx.x;
                float val = data[idx];
                for (int i = 0; i < loops; i++) {{
                    val = sin(val) * cos(val) + tan(val);
                    val = sqrt(abs(val)) + 1.0f;
                }}
                data[idx] = val;
            }}
        """)
        func = mod.get_function("pure_compute_kernel")
        
        while True:
            func(gpu_data, np.int32(loop_limit), block=(1024, 1, 1), grid=(1, 1, 1))
            cuda.Context.synchronize()

    print(f"[+] Starting GPU Compute stressor: {intensity_percentage}% intensity.")
    p = multiprocessing.Process(target=_gpu_comp_worker, args=(intensity_percentage,))
    p.daemon = True
    p.start()
    return [p]

# ==========================================
# 4. VRAM & PCIe BANDWIDTH STRESSOR
# ==========================================
def stress_vram_bandwidth(intensity_percentage):
    """
    Launches a dedicated process pushing massive byte matrices back and forth 
    over the PCIe bus into VRAM channels. GPU math compute remains idle.
    """
    if intensity_percentage <= 0:
        return []
    if not PYCUDA_AVAILABLE:
        print("[!] VRAM Stress requested but PyCUDA/CUDA is not available.")
        return []

    def _vram_worker(intensity):
        # Re-initialize CUDA inside the child process context
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        size = int(64000000 * (intensity / 100.0))
        if size < 1000: size = 1000
        
        host_data = np.random.randn(size).astype(np.float32)
        gpu_data = cuda.mem_alloc(host_data.nbytes)
        
        while True:
            cuda.memcpy_htod(gpu_data, host_data) # Host -> VRAM (Write)
            cuda.memcpy_dtoh(host_data, gpu_data) # VRAM -> Host (Read)
            if intensity < 90:
                time.sleep(0.002)

    print(f"[+] Starting VRAM Bandwidth stressor: {intensity_percentage}% intensity.")
    p = multiprocessing.Process(target=_vram_worker, args=(intensity_percentage,))
    p.daemon = True
    p.start()
    return [p]

# ==========================================
# UNIFIED ORCHESTRATION INTERFACE
# ==========================================
def run_experiment(cpu_pct=0, ram_pct=0, gpu_pct=0, vram_pct=0, duration_seconds=30):
    """
    Main function to execute the cross-subsystem contention experiment.
    All percentage arguments range from 0 (off) to 100 (maximum).
    """
    print("=" * 60)
    print(f"LAUNCHING EXPERIMENT (Duration: {duration_seconds}s)")
    print("=" * 60)
    
    active_processes = []
    
    # Fire up the user-defined workload composition
    active_processes.extend(stress_cpu(cpu_pct))
    active_processes.extend(stress_ram_bandwidth(ram_pct))
    active_processes.extend(stress_gpu_compute(gpu_pct))
    active_processes.extend(stress_vram_bandwidth(vram_pct))
    
    print(f"\n[!] Workload engaged. Running for {duration_seconds} seconds...")
    try:
        time.sleep(duration_seconds)
        print("\n[+] Target duration reached. Terminating stressors clean...")
    except KeyboardInterrupt:
        print("\n[!] Experiment interrupted prematurely by user. Cleaning up...")
    finally:
        # Safely shut down all background workers to free system resources
        for p in active_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("[+] All systems returned to idle state.")
        print("=" * 60)

if __name__ == "__main__":
    # Check if dependencies are missing and warm the user
    if not PYCUDA_AVAILABLE:
        print("[Warning] PyCUDA or CUDA toolkit not detected. GPU/VRAM functions will be skipped.")
        print("Fix: Install NVIDIA CUDA Toolkit and run: pip install pycuda\n")
        
    # EXAMPLE EXPERIMENT CALL:
    # Stresses CPU at 50%, System RAM at 80%, GPU Cores at 90%, and forces 40% VRAM throughput.
    run_experiment(
        cpu_pct=50, 
        ram_pct=60, 
        gpu_pct=90, 
        vram_pct=60, 
        duration_seconds=10000
    )
