"""
Profile a small LLM using PyTorch Profiler with FSDP (Fully Sharded Data Parallel) across multiple GPUs.
Uses PyTorch's native FSDP with FULL_SHARD strategy for distributed inference with rich communication patterns.

FSDP FULL_SHARD shards model parameters across all GPUs and uses AllGather collectives
during the forward pass, making it ideal for studying distributed communication patterns.

Launch with: torchrun --nproc_per_node=4 profile_llm_parallel_pt_fsdp.py
"""

import torch
import torch.distributed as dist
import torch.profiler
from torch.profiler import ExecutionTraceObserver
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "tinyllama"           # Options: 'phi-2', 'tinyllama', 'phi-3-mini', 'llama-2-13b', 'mistral-7b', or HF model path
                                   # Note: Using tinyllama (1.1B) - phi-2 has FSDP compatibility issues
                                   # TinyLlama works perfectly with FSDP and shows same communication patterns
INPUT_PROMPT = "Tell me about the corpus callosum."

# FSDP Configuration
STRATEGY = "fsdp_full"  # FULL_SHARD - shards parameters across all GPUs

# Generation parameters
WARMUP_TOKENS = 5              # Generate this many tokens before profiling
PROFILED_TOKENS = 1             # Profile this many token generations
COOLDOWN_TOKENS = 10           # Generate this many more after profiling (to see response)

# Profiling configuration
PROFILE_ALL_RANKS = True       # Profile all GPUs (True) or just rank 0 (False)

OUTPUT_DIR = './traces_parallel_pt_fsdp'
CHAKRA_ET_OUTPUT = './CPU_trace_parallel_pt_fsdp'

# ============================================================================

model_map = {
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}

# Allow direct HuggingFace model paths
if MODEL_NAME in model_map:
    hf_model_name = model_map[MODEL_NAME]
else:
    hf_model_name = MODEL_NAME  # Assume it's a direct HF path

# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed():
    """Initialize PyTorch distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    return rank, world_size

rank, world_size = setup_distributed()

if rank == 0:
    print(f"PyTorch FSDP Distributed Setup:")
    print(f"  World size: {world_size}")
    print(f"  Strategy: {STRATEGY} (FULL_SHARD)")
    
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. FSDP requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================================
# MODEL LOADING
# ============================================================================

if rank == 0:
    print(f"\nLoading {hf_model_name} with FSDP FULL_SHARD")

# Load tokenizer (all ranks)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU first, then wrap with FSDP
# FSDP will handle moving and sharding to GPU
if rank == 0:
    print(f"  Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

if rank == 0:
    print(f"  Wrapping model with FSDP...")

# Wrap model with FSDP
# FULL_SHARD strategy:
# - Shards parameters, gradients, and optimizer states across all GPUs
# - Each GPU owns 1/N of the parameters
# - Forward pass: AllGather parameters before each layer, then discard
# - This creates rich communication patterns ideal for profiling
#
# Note: Not using use_orig_params=True due to compatibility issues with phi-2
# Generation still works, just uses FSDP's flattened parameters
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
)

# Set to evaluation mode
model.eval()

if rank == 0:
    print(f"  Model initialized with FSDP FULL_SHARD")
    print(f"  Parameters are sharded across {world_size} GPUs")
    print(f"  Each GPU owns ~1/{world_size} of the parameters")
    print(f"  Forward pass: AllGather on every layer")

# ============================================================================
# INPUT PREPARATION
# ============================================================================

# Prepare inputs and move to current rank's device
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(f"cuda:{rank}") for k, v in inputs.items()}

# Create output directory (only rank 0)
if rank == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGeneration plan:")
    print(f"  1. Warmup: Generate {WARMUP_TOKENS} tokens (not profiled)")
    print(f"  2. Profile: Generate {PROFILED_TOKENS} token(s) (PROFILED)")
    print(f"  3. Cooldown: Generate {COOLDOWN_TOKENS} more tokens (not profiled)")
    print(f"  Total tokens: {WARMUP_TOKENS + PROFILED_TOKENS + COOLDOWN_TOKENS}")
    print(f"\nProfiling configuration:")
    print(f"  Profile all ranks: {PROFILE_ALL_RANKS}")
    if PROFILE_ALL_RANKS:
        print(f"  Output: {world_size} trace files (one per GPU)")
    else:
        print(f"  Output: 1 trace file (rank 0 only)")

# ============================================================================
# PHASE 1: WARMUP GENERATION (NOT PROFILED)
# ============================================================================

if rank == 0:
    print(f"\nPhase 1: Warmup generation ({WARMUP_TOKENS} tokens)...")

# All ranks participate in generation
with torch.no_grad():
    warmup_ids = model.generate(
        **inputs,
        max_new_tokens=WARMUP_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

if rank == 0:
    warmup_text = tokenizer.decode(warmup_ids[0], skip_special_tokens=True)
    print(f"  After warmup: {warmup_text}")

# ============================================================================
# PHASE 2: PROFILED GENERATION
# ============================================================================

if rank == 0:
    print(f"\nPhase 2: Profiling {PROFILED_TOKENS} token(s)...")
    print(f"  Rank {rank}: Starting profiling...")
elif PROFILE_ALL_RANKS:
    print(f"  Rank {rank}: Starting profiling...")

# Decide whether this rank should profile
should_profile = PROFILE_ALL_RANKS or (rank == 0)

if should_profile:
    # Setup Chakra ET observer
    et_observer = ExecutionTraceObserver()
    et_observer.register_callback(f"{CHAKRA_ET_OUTPUT}_rank{rank}.json")
    et_observer.start()
    
    # Setup PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUTPUT_DIR}/rank{rank}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True,
            enable_cuda_sync_events=True  # Critical for accurate multi-GPU timing
        )
    ) as prof:
        with torch.profiler.record_function("ProfilerStep#0"):
            with torch.no_grad():
                profiled_ids = model.generate(
                    input_ids=warmup_ids,
                    max_new_tokens=PROFILED_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        prof.step()
    
    # Stop Chakra ET observer
    et_observer.stop()
    output_file = et_observer.get_output_file_path()
    et_observer.unregister_callback()
    
    if rank == 0:
        profiled_text = tokenizer.decode(profiled_ids[0], skip_special_tokens=True)
        print(f"  After profiling: {profiled_text}")
        print(f"  Rank {rank}: Profiling complete")
    else:
        print(f"  Rank {rank}: Profiling complete")
else:
    # Non-profiling ranks still participate in generation
    with torch.no_grad():
        profiled_ids = model.generate(
            input_ids=warmup_ids,
            max_new_tokens=PROFILED_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

# ============================================================================
# PHASE 3: COOLDOWN GENERATION (NOT PROFILED)
# ============================================================================

if rank == 0:
    print(f"\nPhase 3: Cooldown generation ({COOLDOWN_TOKENS} tokens)...")

with torch.no_grad():
    final_ids = model.generate(
        input_ids=profiled_ids,
        max_new_tokens=COOLDOWN_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

if rank == 0:
    final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    print(f"  Final output: {final_text}")

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

if rank == 0:
    print(f"\n{'='*60}")
    print(f"Profiling complete!")
    print(f"{'='*60}")
    print(f"Strategy: FSDP FULL_SHARD")
    print(f"  - Parameters sharded across {world_size} GPUs")
    print(f"  - Forward pass: AllGather on every layer")
    print(f"  - All GPUs active during generation")
    print(f"\nTrace files saved:")
    if PROFILE_ALL_RANKS:
        for r in range(world_size):
            print(f"  Rank {r}: {OUTPUT_DIR}/rank{r}/")
            print(f"  Rank {r}: {CHAKRA_ET_OUTPUT}_rank{r}.json")
    else:
        print(f"  Rank 0: {OUTPUT_DIR}/rank0/")
        print(f"  Rank 0: {CHAKRA_ET_OUTPUT}_rank0.json")
    
    print(f"\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        if allocated > 0:
            print(f"  GPU {i}: {allocated:.2f} GB")
    
    print(f"\nExpected communication patterns in traces:")
    print(f"  - AllGather operations on every transformer layer")
    print(f"  - ReduceScatter during backward (if training)")
    print(f"  - All {world_size} GPUs computing in parallel")
    print(f"  - Synchronization points at layer boundaries")
    print(f"{'='*60}")

# Cleanup distributed
dist.destroy_process_group()

