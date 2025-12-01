"""
Profile a small LLM using PyTorch Profiler with Tensor Parallelism (TP) across multiple GPUs.
Uses PyTorch's native tensor parallel APIs to split individual layers across GPUs.

Tensor Parallelism splits attention heads and MLP layers across GPUs, creating
AllReduce collectives after each attention and MLP block (2× per transformer layer).
This creates the most communication-heavy pattern, ideal for studying distributed patterns.

Launch with: torchrun --nproc_per_node=4 profile_llm_parallel_pt_tp.py

Note: Requires PyTorch 2.0+ with tensor parallel support.
"""

import torch
import torch.distributed as dist
import torch.profiler
from torch.profiler import ExecutionTraceObserver
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "phi-2"               # Options: 'phi-2', 'tinyllama', 'phi-3-mini', 'llama-2-13b', 'mistral-7b'
                                   # Using phi-2 (2.7B) for fair comparison across all parallelism strategies
INPUT_PROMPT = "Tell me about the corpus callosum."

# Model architecture (needed for layer-specific wrapping)
# This determines which layer names to use for parallelization
MODEL_ARCH = "phi"  # Options: "mistral", "llama", "phi"

# Generation parameters
WARMUP_TOKENS = 5              # Generate this many tokens before profiling
PROFILED_TOKENS = 1             # Profile this many token generations
COOLDOWN_TOKENS = 10           # Generate this many more after profiling (to see response)

# Profiling configuration
PROFILE_ALL_RANKS = True       # Profile all GPUs (True) or just rank 0 (False)

OUTPUT_DIR = './traces_parallel_pt_tp'
CHAKRA_ET_OUTPUT = './CPU_trace_parallel_pt_tp'

# ============================================================================

model_map = {
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
}

# Architecture to model name mapping
arch_map = {
    "mistral": ["mistral-7b"],
    "llama": ["llama-2-7b", "llama-2-13b", "tinyllama"],
    "phi": ["phi-2", "phi-3-mini"],
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
    print(f"PyTorch Tensor Parallelism Distributed Setup:")
    print(f"  World size: {world_size}")
    print(f"  Strategy: Tensor Parallelism (TP)")
    print(f"  Model architecture: {MODEL_ARCH}")
    
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Tensor Parallelism requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================================
# TENSOR PARALLEL SETUP
# ============================================================================

try:
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
    from torch.distributed.device_mesh import init_device_mesh
    TP_AVAILABLE = True
except ImportError:
    TP_AVAILABLE = False
    if rank == 0:
        print("\n" + "="*60)
        print("ERROR: Tensor Parallelism not available!")
        print("Requires PyTorch 2.0+ with tensor parallel support.")
        print("Please upgrade PyTorch: pip install torch>=2.0.0")
        print("="*60)
    dist.destroy_process_group()
    exit(1)

# Create device mesh for tensor parallelism
device_mesh = init_device_mesh("cuda", (world_size,))

if rank == 0:
    print(f"\nDevice mesh created: {world_size} GPUs")

# ============================================================================
# MODEL LOADING
# ============================================================================

if rank == 0:
    print(f"\nLoading {hf_model_name} with Tensor Parallelism")

# Load tokenizer (all ranks)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU first, then move to GPU
if rank == 0:
    print(f"  Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

if rank == 0:
    print(f"  Moving model to GPU {rank}...")

model = model.to(f"cuda:{rank}")

# ============================================================================
# APPLY TENSOR PARALLELISM
# ============================================================================

if rank == 0:
    print(f"  Applying Tensor Parallelism to layers...")
    print(f"  This will split attention heads and MLP across {world_size} GPUs")

# Define parallelization plans for different architectures
# ColwiseParallel: splits columns (output dimension) - no communication needed
# RowwiseParallel: splits rows (input dimension) - requires AllReduce after

def get_attention_plan(arch):
    """Get tensor parallel plan for attention layers."""
    if arch in ["mistral", "llama"]:
        return {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),  # AllReduce here
        }
    elif arch == "phi":
        # Phi models use different attention structure
        return {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "dense": RowwiseParallel(),  # AllReduce here
        }
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def get_mlp_plan(arch):
    """Get tensor parallel plan for MLP layers."""
    if arch in ["mistral", "llama"]:
        return {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),  # AllReduce here
        }
    elif arch == "phi":
        # Phi models use different MLP structure
        return {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),  # AllReduce here
        }
    else:
        raise ValueError(f"Unknown architecture: {arch}")

attention_plan = get_attention_plan(MODEL_ARCH)
mlp_plan = get_mlp_plan(MODEL_ARCH)

# Apply tensor parallelism to each transformer layer
# This splits the model horizontally across GPUs
num_layers = len(model.model.layers)

if rank == 0:
    print(f"  Parallelizing {num_layers} transformer layers...")

for layer_id, layer in enumerate(model.model.layers):
    try:
        # Parallelize attention
        parallelize_module(
            layer.self_attn,
            device_mesh,
            attention_plan
        )
        
        # Parallelize MLP
        parallelize_module(
            layer.mlp,
            device_mesh,
            mlp_plan
        )
        
        if rank == 0 and (layer_id == 0 or layer_id == num_layers - 1):
            print(f"    Layer {layer_id}: Parallelized ✓")
    except Exception as e:
        if rank == 0:
            print(f"    Layer {layer_id}: Failed - {e}")
            print(f"    This might be due to architecture mismatch.")
            print(f"    Please check MODEL_ARCH setting.")
        dist.destroy_process_group()
        exit(1)

# Set to evaluation mode
model.eval()

if rank == 0:
    print(f"\n  Model initialized with Tensor Parallelism")
    print(f"  Attention heads split across {world_size} GPUs")
    print(f"  MLP layers split across {world_size} GPUs")
    print(f"  Expected: 2 AllReduces per layer × {num_layers} layers = {2 * num_layers} AllReduces")

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
    print(f"  Note: model.generate() may not work perfectly with TP")
    print(f"  If it fails, we'll need to implement manual generation loop")

# Try using model.generate() - it may or may not work with TP
try:
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
        print(f"  ✓ model.generate() works with TP!")
    
    GENERATE_WORKS = True

except Exception as e:
    if rank == 0:
        print(f"  ✗ model.generate() failed with TP: {e}")
        print(f"  Falling back to manual generation loop...")
    
    GENERATE_WORKS = False
    
    # Manual generation loop
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        for _ in range(WARMUP_TOKENS):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    warmup_ids = input_ids
    
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
                if GENERATE_WORKS:
                    profiled_ids = model.generate(
                        input_ids=warmup_ids,
                        max_new_tokens=PROFILED_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    # Manual generation
                    input_ids = warmup_ids
                    for _ in range(PROFILED_TOKENS):
                        outputs = model(input_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                    profiled_ids = input_ids
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
        if GENERATE_WORKS:
            profiled_ids = model.generate(
                input_ids=warmup_ids,
                max_new_tokens=PROFILED_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            input_ids = warmup_ids
            for _ in range(PROFILED_TOKENS):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            profiled_ids = input_ids

# ============================================================================
# PHASE 3: COOLDOWN GENERATION (NOT PROFILED)
# ============================================================================

if rank == 0:
    print(f"\nPhase 3: Cooldown generation ({COOLDOWN_TOKENS} tokens)...")

with torch.no_grad():
    if GENERATE_WORKS:
        final_ids = model.generate(
            input_ids=profiled_ids,
            max_new_tokens=COOLDOWN_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        input_ids = profiled_ids
        for _ in range(COOLDOWN_TOKENS):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        final_ids = input_ids

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
    print(f"Strategy: Tensor Parallelism (TP)")
    print(f"  - Attention heads split across {world_size} GPUs")
    print(f"  - MLP layers split across {world_size} GPUs")
    print(f"  - Forward pass: AllReduce after attention + AllReduce after MLP")
    print(f"  - Total: {2 * num_layers} AllReduces per forward pass")
    print(f"  - All GPUs compute in parallel")
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
    print(f"  - AllReduce after every attention block")
    print(f"  - AllReduce after every MLP block")
    print(f"  - Most communication-heavy strategy!")
    print(f"  - All {world_size} GPUs computing in parallel")
    print(f"{'='*60}")

# Cleanup distributed
dist.destroy_process_group()

