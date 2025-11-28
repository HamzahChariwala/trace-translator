"""
Test whether PyTorch Profiler is process-scoped or device-scoped.

Two tests:
1. Parent profiler wrapping child processes - does it capture child activity?
2. Separate profilers per process - baseline to compare against
"""

import torch
import torch.profiler
from torch.profiler import ExecutionTraceObserver
import torch.multiprocessing as mp
import os
import glob
import shutil


# Configuration
OUTPUT_BASE = './test_traces/profiler_scope_test'
GPU_ID = 0  # All processes will use THIS SAME GPU


def worker_no_profiler(rank, world_size):
    """
    Worker that does NOT have its own profiler.
    Tests if parent's profiler captures this activity.
    """
    pid = os.getpid()
    device = f'cuda:{GPU_ID}'
    
    print(f"[Worker {rank}] PID: {pid}, Device: {device}, NO profiler")
    
    # Create tensor and do work
    tensor = torch.randn(500, 500, device=device)
    
    with torch.profiler.record_function(f"Worker{rank}_NoProfiler"):
        result = tensor @ tensor.T
        torch.cuda.synchronize(device=GPU_ID)
    
    print(f"[Worker {rank}] Work complete, PID: {pid}")


def worker_with_profiler(rank, world_size):
    """
    Worker that HAS its own profiler.
    Baseline test - each process exports independently.
    """
    pid = os.getpid()
    device = f'cuda:{GPU_ID}'
    
    # Separate output directory per process
    output_dir = f"{OUTPUT_BASE}/test2_separate_profilers/process_{rank}"
    chakra_et_path = f"{OUTPUT_BASE}/test2_separate_profilers/process_{rank}_CPU_trace"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(chakra_et_path), exist_ok=True)
    
    print(f"[Worker {rank}] PID: {pid}, Device: {device}, WITH profiler")
    
    # Create tensor
    tensor = torch.randn(500, 500, device=device)
    
    # Setup profiling IN THIS PROCESS
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
        with torch.profiler.record_function(f"Worker{rank}_WithProfiler"):
            result = tensor @ tensor.T
            torch.cuda.synchronize(device=GPU_ID)
        
        prof.step()
    
    et_observer.stop()
    et_observer.unregister_callback()
    
    print(f"[Worker {rank}] Profiling complete, PID: {pid}")


def test1_parent_profiler():
    """
    TEST 1: Parent process has profiler, spawns child processes without profilers.
    Question: Does parent's profiler capture child process activity?
    """
    print("="*70)
    print("TEST 1: Parent Profiler Wrapping Child Processes")
    print("="*70)
    print(f"Main PID: {os.getpid()}")
    print(f"All processes will use GPU {GPU_ID}")
    
    # List files BEFORE test
    files_before = list_all_files(OUTPUT_BASE)
    print(f"\nFiles before test: {len(files_before)}")
    
    output_dir = f"{OUTPUT_BASE}/test1_parent_profiler"
    chakra_et_path = f"{OUTPUT_BASE}/test1_parent_profiler/CPU_trace_test1"
    os.makedirs(output_dir, exist_ok=True)
    
    world_size = 2
    print(f"\nStarting parent profiler, then spawning {world_size} child processes...\n")
    
    # Setup profiling in PARENT
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
        with torch.profiler.record_function("ParentProfiler_SpawningChildren"):
            # Spawn child processes (they don't have their own profilers)
            mp.spawn(
                worker_no_profiler,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
        
        prof.step()
    
    et_observer.stop()
    et_observer.unregister_callback()
    
    print("\nParent profiler stopped. Checking traces...\n")
    
    # List files AFTER test
    files_after = list_all_files(OUTPUT_BASE)
    new_files = files_after - files_before
    
    print("="*70)
    print("TEST 1 RESULTS")
    print("="*70)
    print(f"New files created: {len(new_files)}")
    
    kineto_files = []
    cpu_files = []
    
    for f in sorted(new_files):
        full_path = os.path.join(OUTPUT_BASE, f)
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
        
        if f.endswith('.pt.trace.json'):
            kineto_files.append(f)
        elif 'CPU_trace' in f and f.endswith('.json'):
            cpu_files.append(f)
    
    print(f"\nKineto traces: {len(kineto_files)}")
    print(f"CPU traces: {len(cpu_files)}")
    
    print("\nINTERPRETATION:")
    if len(kineto_files) == 1 and len(cpu_files) == 1:
        print("  → Parent profiler captured activity (1 Kineto + 1 CPU trace)")
        print("  → But did it capture child process activity? Need to inspect trace contents.")
    else:
        print(f"  → Unexpected: {len(kineto_files)} Kineto, {len(cpu_files)} CPU traces")


def test2_separate_profilers():
    """
    TEST 2: Each child process has its own profiler (baseline).
    Expected: N processes → N traces (since each exports independently).
    """
    print("\n" + "="*70)
    print("TEST 2: Separate Profilers Per Process (Baseline)")
    print("="*70)
    print(f"Main PID: {os.getpid()}")
    
    # List files BEFORE test
    files_before = list_all_files(OUTPUT_BASE)
    print(f"\nFiles before test: {len(files_before)}")
    
    world_size = 2
    print(f"\nSpawning {world_size} processes, each with its own profiler...\n")
    
    mp.spawn(
        worker_with_profiler,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    print("\nAll processes complete. Checking traces...\n")
    
    # List files AFTER test
    files_after = list_all_files(OUTPUT_BASE)
    new_files = files_after - files_before
    
    print("="*70)
    print("TEST 2 RESULTS")
    print("="*70)
    print(f"New files created: {len(new_files)}")
    
    kineto_files = []
    cpu_files = []
    
    for f in sorted(new_files):
        full_path = os.path.join(OUTPUT_BASE, f)
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
        
        if f.endswith('.pt.trace.json'):
            kineto_files.append(f)
        elif 'CPU_trace' in f and f.endswith('.json'):
            cpu_files.append(f)
    
    print(f"\nKineto traces: {len(kineto_files)}")
    print(f"CPU traces: {len(cpu_files)}")
    
    print("\nINTERPRETATION:")
    if len(kineto_files) == world_size and len(cpu_files) == world_size:
        print(f"  → Each process generated its own traces (baseline confirmed)")
        print(f"  → This is expected since each process has separate profiler instance")
    else:
        print(f"  → Unexpected: {len(kineto_files)} Kineto, {len(cpu_files)} CPU traces (expected {world_size} each)")


def list_all_files(base_dir):
    """Recursively list all files in a directory"""
    if not os.path.exists(base_dir):
        return set()
    
    all_files = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dir)
            all_files.add(rel_path)
    return all_files


def clean_output_dir():
    """Clean the output directory before running tests"""
    if os.path.exists(OUTPUT_BASE):
        print(f"Cleaning output directory: {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)
    os.makedirs(OUTPUT_BASE, exist_ok=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this test")
    
    print("\n" + "#"*70)
    print("#  Profiler Scope Test")
    print("#  Testing whether profiler is process-scoped or device-scoped")
    print("#"*70)
    
    # Clean before starting
    clean_output_dir()
    
    # Run TEST 1
    test1_parent_profiler()
    
    # Wait for user to review before continuing
    print("\n" + "="*70)
    print("TEST 1 COMPLETE - Review the results above")
    print("="*70)
    input("Press Enter to continue to TEST 2 (this will clean TEST 1 traces)...")
    
    # Clean before TEST 2
    clean_output_dir()
    
    # Run TEST 2
    test2_separate_profilers()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("TEST 1: If parent profiler ONLY captures parent activity:")
    print("  → Profiler is PROCESS-SCOPED (Hypothesis 1A)")
    print("TEST 2: Baseline showing separate profilers → separate traces")
    print("\nFor your use case (single process, multi-GPU model):")
    print("  - If 1A is true: Need to post-process single trace by device ID")
    print("  - Or: Try separate profiler contexts per GPU (needs testing)")
    print("="*70)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

