"""
Step 1: Verify what creates separate OS processes vs threads.
This script tests different parallelism methods and prints their PIDs.

Requirements: Only PyTorch (already in environment.yml)
Platform: Works on Ubuntu/Linux, macOS, Windows
"""

import torch
import torch.multiprocessing as mp
import os
import sys


def print_separator(title):
    """Print a visual separator"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_single_process_multi_gpu():
    """Test 1: Single process using multiple GPUs manually"""
    print_separator("TEST 1: Single Process, Multiple GPUs (Manual Placement)")
    
    print(f"Main process PID: {os.getpid()}")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU tests")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Create tensors on different GPUs
    print("\nCreating tensors on different GPUs...")
    for i in range(min(num_gpus, 4)):  # Test up to 4 GPUs
        tensor = torch.randn(100, 100, device=f'cuda:{i}')
        print(f"  GPU {i}: tensor created, PID still {os.getpid()}")
    
    print(f"\nConclusion: All operations in PID {os.getpid()} (single process)")


def worker_multiprocessing(rank, world_size):
    """Worker function for torch.multiprocessing.spawn()"""
    pid = os.getpid()
    print(f"  [Rank {rank}] PID: {pid}")
    
    if torch.cuda.is_available():
        # Each worker uses a different GPU
        gpu_id = rank % torch.cuda.device_count()
        tensor = torch.randn(100, 100, device=f'cuda:{gpu_id}')
        print(f"  [Rank {rank}] Using GPU {gpu_id}, PID: {pid}")


def test_multiprocessing_spawn():
    """Test 2: torch.multiprocessing.spawn()"""
    print_separator("TEST 2: torch.multiprocessing.spawn()")
    
    main_pid = os.getpid()
    print(f"Main process PID: {main_pid}")
    
    world_size = 2  # Spawn 2 processes
    print(f"Spawning {world_size} worker processes...\n")
    
    try:
        mp.spawn(
            worker_multiprocessing,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print(f"\nMain process PID after spawn: {os.getpid()}")
        print("Conclusion: Each spawned worker has a DIFFERENT PID (separate processes)")
    except Exception as e:
        print(f"Error during spawn: {e}")


def test_dataparallel():
    """Test 3: Check if DataParallel creates new processes"""
    print_separator("TEST 3: torch.nn.DataParallel")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping")
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need at least 2 GPUs for DataParallel test")
        return
    
    print(f"Process PID: {os.getpid()}")
    print(f"Number of GPUs: {num_gpus}")
    
    # Simple model
    model = torch.nn.Linear(100, 100)
    
    # Wrap with DataParallel
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    print(f"After DataParallel initialization, PID: {os.getpid()}")
    
    # Run forward pass
    input_tensor = torch.randn(64, 100).cuda()
    output = model(input_tensor)
    
    print(f"After forward pass, PID: {os.getpid()}")
    print("Conclusion: DataParallel uses SAME PID (single process, likely uses threads)")


def test_threading():
    """Test 4: Python threading for comparison"""
    print_separator("TEST 4: Python threading.Thread (for comparison)")
    
    import threading
    
    main_pid = os.getpid()
    print(f"Main thread PID: {main_pid}")
    print(f"Main thread ID: {threading.get_ident()}")
    
    def thread_worker(thread_id):
        pid = os.getpid()
        tid = threading.get_ident()
        print(f"  [Thread {thread_id}] PID: {pid}, Thread ID: {tid}")
    
    threads = []
    print("\nCreating 2 threads...\n")
    for i in range(2):
        t = threading.Thread(target=thread_worker, args=(i,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print(f"\nConclusion: All threads share SAME PID {main_pid} (same process)")


def main():
    print("\n" + "#"*70)
    print("#  Process ID Verification Script")
    print("#  Testing different parallelism methods in PyTorch")
    print("#"*70)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Run all tests
    test_single_process_multi_gpu()
    
    # Set multiprocessing start method (required for CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    test_multiprocessing_spawn()
    test_dataparallel()
    test_threading()
    
    # Summary
    print_separator("SUMMARY")
    print("Different PIDs = Separate OS processes")
    print("Same PID, different thread IDs = Same process, different threads")
    print("\nExpected results:")
    print("  - Single process multi-GPU: Same PID")
    print("  - torch.multiprocessing.spawn(): Different PIDs")
    print("  - DataParallel: Same PID (uses threads)")
    print("  - Python threading: Same PID, different thread IDs")
    print("\nFor profiling hypothesis 1A:")
    print("  If profiler is process-scoped, methods with different PIDs")
    print("  should generate separate trace files.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

