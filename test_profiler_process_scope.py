"""
Test whether PyTorch Profiler is process-scoped or device-scoped.

Critical Test: Multiple processes on SAME GPU
- If we get N traces for N processes → profiler is PROCESS-SCOPED (supports 1A)
- If we get 1 trace → profiler is DEVICE-SCOPED (falsifies 1A)
"""

import torch
import torch.profiler
from torch.profiler import ExecutionTraceObserver
import torch.multiprocessing as mp
import os
import glob


# Configuration
OUTPUT_BASE = './test_traces/profiler_scope_test'
GPU_ID = 0  # All processes will use THIS SAME GPU


def profile_worker(rank, world_size):
    """
    Worker function that runs in a separate process.
    Each process profiles operations on the SAME GPU.
    """
    pid = os.getpid()
    device = f'cuda:{GPU_ID}'
    
    # Separate output directory per process
    output_dir = f"{OUTPUT_BASE}/process_{rank}"
    chakra_et_path = f"{OUTPUT_BASE}/process_{rank}_et"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(chakra_et_path), exist_ok=True)
    
    print(f"[Process {rank}] PID: {pid}, Device: {device}")
    
    # Create tensor on the shared GPU
    tensor = torch.randn(500, 500, device=device)
    
    # Setup profiling
    et_observer = ExecutionTraceObserver()
    et_observer.register_callback(chakra_et_path + ".json")
    et_observer.start()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True,
            enable_cuda_sync_events=True
        )
    ) as prof:
        with torch.profiler.record_function(f"Process{rank}_Operation"):
            # Do some work
            result = tensor @ tensor.T
            torch.cuda.synchronize(device=GPU_ID)
        
        prof.step()
    
    et_observer.stop()
    et_observer.unregister_callback()
    
    print(f"[Process {rank}] Profiling complete, PID: {pid}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this test")
    
    print("="*70)
    print("TEST: Multiple Processes, Single GPU (Process-Scoped vs Device-Scoped)")
    print("="*70)
    print(f"Main PID: {os.getpid()}")
    print(f"All processes will use GPU {GPU_ID}")
    
    # Clean output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Spawn 2 processes that both use the same GPU
    world_size = 2
    print(f"\nSpawning {world_size} processes (both using GPU {GPU_ID})...\n")
    
    mp.spawn(
        profile_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    # Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    total_kineto_traces = 0
    total_et_traces = 0
    
    for rank in range(world_size):
        output_dir = f"{OUTPUT_BASE}/process_{rank}"
        
        # Count Kineto traces
        if os.path.exists(output_dir):
            kineto_files = glob.glob(f"{output_dir}/*.pt.trace.json")
            total_kineto_traces += len(kineto_files)
            print(f"Process {rank}: {len(kineto_files)} Kineto trace(s)")
            for f in kineto_files:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                print(f"  - {os.path.basename(f)} ({size_mb:.2f} MB)")
        
        # Count Chakra ET traces
        et_file = f"{OUTPUT_BASE}/process_{rank}_et.json"
        if os.path.exists(et_file):
            total_et_traces += 1
            size_mb = os.path.getsize(et_file) / (1024 * 1024)
            print(f"Process {rank}: 1 Chakra ET trace ({size_mb:.2f} MB)")
    
    print(f"\n{'='*70}")
    print(f"TOTAL Kineto traces: {total_kineto_traces}")
    print(f"TOTAL Chakra ET traces: {total_et_traces}")
    print(f"Number of processes: {world_size}")
    print(f"GPU used by all processes: {GPU_ID}")
    print(f"{'='*70}")
    
    # Interpret results
    print("\nINTERPRETATION:")
    if total_kineto_traces == world_size:
        print(f"✓ Got {total_kineto_traces} Kineto traces for {world_size} processes")
        print("  → Profiler is PROCESS-SCOPED (supports Hypothesis 1A)")
    elif total_kineto_traces == 1:
        print(f"✗ Got only 1 Kineto trace for {world_size} processes")
        print("  → Profiler is DEVICE-SCOPED or aggregates traces (falsifies 1A)")
    else:
        print(f"? Got {total_kineto_traces} traces (unexpected)")
    
    if total_et_traces == world_size:
        print(f"✓ Got {total_et_traces} Chakra ET traces for {world_size} processes")
        print("  → ExecutionTraceObserver is PROCESS-SCOPED")
    elif total_et_traces == 1:
        print(f"✗ Got only 1 Chakra ET trace for {world_size} processes")
        print("  → ExecutionTraceObserver aggregates or is device-scoped")
    else:
        print(f"? Got {total_et_traces} ET traces (unexpected)")
    
    print("="*70)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

