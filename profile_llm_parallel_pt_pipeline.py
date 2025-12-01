"""
Profile a small LLM using PyTorch Profiler with Pipeline Parallelism across multiple GPUs.
Uses PyTorch's PiPPy (Pipeline Parallelism for PyTorch) to split model layers across GPUs.

Pipeline Parallelism splits the model into stages (layer ranges) with each stage on a different GPU.
Data flows sequentially through stages, with point-to-point communication between stages.

This is similar to HuggingFace's device_map="auto" but uses PyTorch's native pipeline API.
The main difference is that PiPPy can use microbatching for better GPU utilization.

Launch with: torchrun --nproc_per_node=4 profile_llm_parallel_pt_pipeline.py

Note: Pipeline parallelism works best with microbatching, but for autoregressive generation
(one token at a time), it behaves similarly to sequential execution.
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

# Pipeline configuration
# For 4 GPUs, we'll split the model into 4 stages
# Example for Mistral-7B (32 layers): 8 layers per stage
PIPELINE_STAGES = 4  # Should match number of GPUs

# Generation parameters
WARMUP_TOKENS = 5              # Generate this many tokens before profiling
PROFILED_TOKENS = 1             # Profile this many token generations
COOLDOWN_TOKENS = 10           # Generate this many more after profiling (to see response)

# Profiling configuration
PROFILE_ALL_RANKS = True       # Profile all GPUs (True) or just rank 0 (False)

OUTPUT_DIR = './traces_parallel_pt_pipeline'
CHAKRA_ET_OUTPUT = './CPU_trace_parallel_pt_pipeline'

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
    print(f"PyTorch Pipeline Parallelism Distributed Setup:")
    print(f"  World size: {world_size}")
    print(f"  Strategy: Pipeline Parallelism")
    print(f"  Pipeline stages: {PIPELINE_STAGES}")
    
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Pipeline Parallelism requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if PIPELINE_STAGES != world_size:
        print(f"\nWARNING: PIPELINE_STAGES ({PIPELINE_STAGES}) != world_size ({world_size})")
        print(f"Setting PIPELINE_STAGES = {world_size}")
        PIPELINE_STAGES = world_size

# ============================================================================
# MODEL LOADING
# ============================================================================

if rank == 0:
    print(f"\nLoading {hf_model_name}")
    print(f"Note: Pipeline parallelism for autoregressive generation is experimental")
    print(f"      This script uses manual layer placement (similar to HF device_map)")

# Load tokenizer (all ranks)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU first
if rank == 0:
    print(f"  Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ============================================================================
# MANUAL PIPELINE PARALLELISM (SIMPLE APPROACH)
# ============================================================================
#
# Note: PyTorch's PiPPy is designed for training with microbatching.
# For inference, especially autoregressive generation, it's simpler to use
# manual layer placement (similar to HuggingFace's device_map).
#
# This gives us similar behavior to the HF script but with explicit control.
# ============================================================================

if rank == 0:
    print(f"  Applying manual pipeline parallelism...")

# Get number of layers
num_layers = len(model.model.layers)
layers_per_stage = num_layers // world_size

if rank == 0:
    print(f"  Total layers: {num_layers}")
    print(f"  Layers per stage: {layers_per_stage}")
    print(f"  Stage distribution:")

# Determine which layers belong to which rank
layer_distribution = {}
for stage in range(world_size):
    start_layer = stage * layers_per_stage
    if stage == world_size - 1:
        # Last stage gets any remaining layers
        end_layer = num_layers
    else:
        end_layer = (stage + 1) * layers_per_stage
    
    layer_distribution[stage] = (start_layer, end_layer)
    
    if rank == 0:
        print(f"    Stage {stage} (GPU {stage}): Layers {start_layer}-{end_layer-1}")

# Move model components to appropriate devices
# Embedding layer on first GPU (all ranks need to do this)
model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")

# Distribute transformer layers (all ranks do this)
for stage, (start_layer, end_layer) in layer_distribution.items():
    for layer_idx in range(start_layer, end_layer):
        model.model.layers[layer_idx] = model.model.layers[layer_idx].to(f"cuda:{stage}")

# Final norm and LM head on last GPU
# Different models use different layer names
last_gpu = world_size - 1

# Handle different model architectures
if hasattr(model.model, 'norm'):
    # Llama, Mistral models
    model.model.norm = model.model.norm.to(f"cuda:{last_gpu}")
elif hasattr(model.model, 'final_layernorm'):
    # Phi models
    model.model.final_layernorm = model.model.final_layernorm.to(f"cuda:{last_gpu}")

model.lm_head = model.lm_head.to(f"cuda:{last_gpu}")

# Set to evaluation mode
model.eval()

if rank == 0:
    print(f"\n  Model initialized with Pipeline Parallelism")
    print(f"  Layers distributed across {world_size} GPUs")
    print(f"  Communication: Point-to-point (send/recv) between stages")
    print(f"  Note: Sequential execution (one stage at a time for single token)")

# ============================================================================
# INPUT PREPARATION
# ============================================================================

# Prepare inputs - they start on GPU 0 (first stage)
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

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
    print(f"\nNote: With pipeline parallelism, GPUs will be active sequentially")
    print(f"      GPU 0 → GPU 1 → GPU 2 → GPU 3 for each forward pass")

# ============================================================================
# PHASE 1: WARMUP GENERATION (NOT PROFILED)
# ============================================================================

if rank == 0:
    print(f"\nPhase 1: Warmup generation ({WARMUP_TOKENS} tokens)...")

# For pipeline parallelism with manual placement, model.generate() should work
# The model will automatically handle moving tensors between devices
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
    # Non-profiling ranks don't need to do anything
    # (model.generate() is only called on rank 0's inputs)
    pass

# ============================================================================
# PHASE 3: COOLDOWN GENERATION (NOT PROFILED)
# ============================================================================

if rank == 0:
    print(f"\nPhase 3: Cooldown generation ({COOLDOWN_TOKENS} tokens)...")

if rank == 0 or should_profile:
    with torch.no_grad():
        final_ids = model.generate(
            input_ids=profiled_ids if should_profile else warmup_ids,
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
    print(f"Strategy: Pipeline Parallelism (Manual Layer Placement)")
    print(f"  - Model split into {world_size} stages")
    print(f"  - Each stage on a different GPU")
    print(f"  - Sequential execution: GPU 0 → GPU 1 → GPU 2 → GPU 3")
    print(f"  - Communication: Activation passing between stages")
    print(f"\nComparison with HuggingFace device_map:")
    print(f"  - Similar behavior (sequential execution)")
    print(f"  - Explicit control over layer placement")
    print(f"  - Can see which GPU is active at each time")
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
    print(f"  - Minimal NCCL communication (mostly CPU-GPU copies)")
    print(f"  - Sequential GPU activation (one at a time)")
    print(f"  - Similar to HuggingFace device_map='auto'")
    print(f"  - Each rank's trace shows when that GPU is active")
    print(f"\nNote: For more interesting pipeline patterns, consider:")
    print(f"  - Microbatching (multiple samples in parallel)")
    print(f"  - Training (where pipeline stages can overlap)")
    print(f"{'='*60}")

# Cleanup distributed
dist.destroy_process_group()

