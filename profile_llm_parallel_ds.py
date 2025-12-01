"""
Profile a small LLM using PyTorch Profiler with DeepSpeed ZeRO-3 across multiple GPUs.
Uses DeepSpeed's ZeRO-3 parameter sharding for distributed inference with rich communication patterns.

ZeRO-3 shards model parameters across all GPUs and uses Broadcast/Gather collectives
during the forward pass, making it ideal for studying distributed communication patterns.

Launch with: deepspeed --num_gpus=4 profile_llm_parallel_ds.py
"""

import torch
import torch.profiler
from torch.profiler import ExecutionTraceObserver
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistral-7b"          # Options: 'phi-2', 'tinyllama', 'phi-3-mini', 'llama-2-13b', 'mistral-7b', or HF model path
INPUT_PROMPT = "Tell me about the corpus callosum."

# DeepSpeed ZeRO-3 Configuration
STRATEGY = "zero3"  # Currently only ZeRO-3 (other stages don't have forward pass communication)

# Generation parameters
WARMUP_TOKENS = 5              # Generate this many tokens before profiling
PROFILED_TOKENS = 1             # Profile this many token generations
COOLDOWN_TOKENS = 10           # Generate this many more after profiling (to see response)

# Profiling configuration
PROFILE_ALL_RANKS = True       # Profile all GPUs (True) or just rank 0 (False)

OUTPUT_DIR = './traces_parallel_ds_zero3'
CHAKRA_ET_OUTPUT = './CPU_trace_parallel_ds_zero3'

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
    print(f"Using custom model path: {MODEL_NAME}")

# ============================================================================
# DEEPSPEED DISTRIBUTED SETUP
# ============================================================================

# DeepSpeed automatically initializes distributed environment via its launcher
# We just need to get our rank information
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)

if local_rank == 0:
    print(f"DeepSpeed Distributed Setup:")
    print(f"  World size: {world_size}")
    print(f"  Strategy: {STRATEGY}")
    
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. DeepSpeed requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ============================================================================
# DEEPSPEED CONFIGURATION
# ============================================================================

def get_ds_config(strategy):
    """
    Generate DeepSpeed configuration for the specified ZeRO stage.
    
    ZeRO-3 shards model parameters across all GPUs:
    - Each GPU owns 1/N of the parameters
    - Forward pass: Broadcast/Gather parameters from owners before each layer
    - This creates rich communication patterns ideal for profiling
    """
    if strategy == "zero3":
        return {
            "train_batch_size": world_size,  # Total batch size across all GPUs
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 3,  # Shard parameters, gradients, and optimizer states
                "offload_param": {
                    "device": "none"  # Keep parameters on GPU for inference
                },
                "offload_optimizer": {
                    "device": "none"  # No optimizer for inference
                },
                "stage3_gather_16bit_weights_on_model_save": True,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
            },
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,  # Disable DeepSpeed's internal profiling
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

ds_config = get_ds_config(STRATEGY)

# ============================================================================
# MODEL LOADING
# ============================================================================

if local_rank == 0:
    print(f"\nLoading {hf_model_name} with DeepSpeed ZeRO-3")

# Load tokenizer (all ranks)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model on CPU first (DeepSpeed will handle GPU placement)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Create a dummy optimizer (required by DeepSpeed even for inference)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Initialize DeepSpeed engine
# This will:
# 1. Shard the model parameters across all GPUs
# 2. Set up communication groups
# 3. Wrap the model for ZeRO-3 operations
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Set to evaluation mode
model_engine.eval()

if local_rank == 0:
    print(f"  Model initialized with DeepSpeed ZeRO-3")
    print(f"  Parameters are sharded across {world_size} GPUs")
    print(f"  Each GPU owns ~1/{world_size} of the parameters")

# ============================================================================
# INPUT PREPARATION
# ============================================================================

# Prepare inputs and move to current rank's device
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(f"cuda:{local_rank}") for k, v in inputs.items()}

# Create output directory (only rank 0)
if local_rank == 0:
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

if local_rank == 0:
    print(f"\nPhase 1: Warmup generation ({WARMUP_TOKENS} tokens)...")

with torch.no_grad():
    # For ZeRO-3, we need to use synced_gpus=True to ensure all ranks participate
    warmup_ids = model_engine.module.generate(
        **inputs,
        max_new_tokens=WARMUP_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        synced_gpus=True,  # Critical for ZeRO-3!
    )

if local_rank == 0:
    warmup_text = tokenizer.decode(warmup_ids[0], skip_special_tokens=True)
    print(f"  After warmup: {warmup_text}")

# ============================================================================
# PHASE 2: PROFILED GENERATION
# ============================================================================

if local_rank == 0:
    print(f"\nPhase 2: Profiling {PROFILED_TOKENS} token(s)...")
    print(f"  Rank {local_rank}: Starting profiling...")
elif PROFILE_ALL_RANKS:
    print(f"  Rank {local_rank}: Starting profiling...")

# Decide whether this rank should profile
should_profile = PROFILE_ALL_RANKS or (local_rank == 0)

if should_profile:
    # Setup Chakra ET observer
    et_observer = ExecutionTraceObserver()
    et_observer.register_callback(f"{CHAKRA_ET_OUTPUT}_rank{local_rank}.json")
    et_observer.start()
    
    # Setup PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUTPUT_DIR}/rank{local_rank}"),
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
                profiled_ids = model_engine.module.generate(
                    input_ids=warmup_ids,
                    max_new_tokens=PROFILED_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    synced_gpus=True,  # Critical for ZeRO-3!
                )
        prof.step()
    
    # Stop Chakra ET observer
    et_observer.stop()
    output_file = et_observer.get_output_file_path()
    et_observer.unregister_callback()
    
    if local_rank == 0:
        profiled_text = tokenizer.decode(profiled_ids[0], skip_special_tokens=True)
        print(f"  After profiling: {profiled_text}")
        print(f"  Rank {local_rank}: Profiling complete")
    else:
        print(f"  Rank {local_rank}: Profiling complete")
else:
    # Non-profiling ranks still participate in generation
    with torch.no_grad():
        profiled_ids = model_engine.module.generate(
            input_ids=warmup_ids,
            max_new_tokens=PROFILED_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            synced_gpus=True,
        )

# ============================================================================
# PHASE 3: COOLDOWN GENERATION (NOT PROFILED)
# ============================================================================

if local_rank == 0:
    print(f"\nPhase 3: Cooldown generation ({COOLDOWN_TOKENS} tokens)...")

with torch.no_grad():
    final_ids = model_engine.module.generate(
        input_ids=profiled_ids,
        max_new_tokens=COOLDOWN_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        synced_gpus=True,
    )

if local_rank == 0:
    final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    print(f"  Final output: {final_text}")

# ============================================================================
# SUMMARY
# ============================================================================

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"Profiling complete!")
    print(f"{'='*60}")
    print(f"Strategy: DeepSpeed ZeRO-3")
    print(f"  - Parameters sharded across {world_size} GPUs")
    print(f"  - Forward pass: Broadcast/Gather on every layer")
    print(f"  - All GPUs active during generation")
    print(f"\nTrace files saved:")
    if PROFILE_ALL_RANKS:
        for rank in range(world_size):
            print(f"  Rank {rank}: {OUTPUT_DIR}/rank{rank}/")
            print(f"  Rank {rank}: {CHAKRA_ET_OUTPUT}_rank{rank}.json")
    else:
        print(f"  Rank 0: {OUTPUT_DIR}/rank0/")
        print(f"  Rank 0: {CHAKRA_ET_OUTPUT}_rank0.json")
    
    print(f"\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        if allocated > 0:
            print(f"  GPU {i}: {allocated:.2f} GB")
    
    print(f"\nExpected communication patterns in traces:")
    print(f"  - Broadcast/Gather operations on every transformer layer")
    print(f"  - All {world_size} GPUs computing in parallel")
    print(f"  - Synchronization points at layer boundaries")
    print(f"{'='*60}")

