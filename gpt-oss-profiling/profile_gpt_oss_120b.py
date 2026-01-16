"""
Profile GPT-OSS-120B using PyTorch Profiler with Pipeline Parallelism across multiple GPUs.
Uses device_map="auto" to automatically distribute layers across GPUs.

GPT-OSS-120B is a Mixture-of-Experts (MoE) model with ~117B parameters total and ~5.1B active per token.
Model size: ~60-65 GB (quantized with MXFP4), requires ~80 GB VRAM for inference.

Pipeline Parallelism distributes different layers across GPUs, with sequential communication
at layer boundaries. This creates GPU-to-GPU data transfers as activations flow through the model.

Launch with: torchrun --nproc_per_node=2 profile_gpt_oss_120b.py

Note: Requires MXFP4 kernels installed (pip install kernels) for efficient quantization.
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

# Choose model size based on available MXFP4 support
# GPT-OSS-120B requires MXFP4 kernels (not yet widely available)
# Without MXFP4, it dequantizes to bf16 (~240GB) which won't fit on 2 H100s

MODEL_NAME = "gpt-oss-120b"                   # RECOMMENDED: Start with smaller model
                                       # Options:
                                       #   "phi-2" (2.7B, ~5GB) - good for testing
                                       #   "llama-2-7b" (7B, ~14GB) - medium size
                                       #   "mistral-7b" (7B, ~14GB) - efficient
                                       #   "gpt-oss-120b" (120B, needs MXFP4 kernels!)
INPUT_PROMPT = "Tell me about the corpus callosum. Give a very extensive description, sparing no detail."

# Model architecture - set based on MODEL_NAME
MODEL_ARCH = "gpt-oss"                     # Options: "phi", "llama", "mistral", "gpt-oss"

# Input sequence configuration
EXACT_INPUT_TOKENS = 500        # Exact number of input tokens (None = use prompt as-is)
                                # If set, will pad or truncate INPUT_PROMPT to this length
INPUT_PAD_TOKEN_ID = None       # Token ID to use for padding (None = use tokenizer default)

# Generation parameters
TOTAL_TOKENS_TO_GENERATE = 1000 # Total number of tokens to generate and profile
WARMUP_TOKENS = 0               # Generate this many tokens before starting profiling (optional warmup)
PROFILE_EACH_TOKEN = True       # Profile each token generation separately (True) or all together (False)

# Profiling configuration
PROFILE_ALL_RANKS = True        # Profile all GPUs (True) or just rank 0 (False)

# Output directories - clearly labeled for CPU vs GPU traces
OUTPUT_DIR_BASE = './GPU_traces_gpt_oss_120b'          # PyTorch GPU traces
CHAKRA_ET_OUTPUT_BASE = './CPU_traces_gpt_oss_120b'    # Chakra ET CPU traces

# ============================================================================

model_map = {
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "gpt-oss-120b": "openai/gpt-oss-120b",  # Official HuggingFace path for GPT-OSS-120B
}

# Architecture to model name mapping
arch_map = {
    "mistral": ["mistral-7b"],
    "llama": ["llama-2-7b", "llama-2-13b", "tinyllama"],
    "phi": ["phi-2", "phi-3-mini"],
    "gpt-oss": ["gpt-oss-120b"],
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
    print(f"  Strategy: Pipeline Parallelism (device_map='auto')")
    print(f"  Model: GPT-OSS-120B (MoE, ~117B params total, ~5.1B active)")
    print(f"  Model size: ~60-65 GB (quantized)")
    print(f"  Model architecture: {MODEL_ARCH}")
    
    # Check GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Pipeline Parallelism requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"         Memory: {mem_gb:.1f} GB")
    
    print(f"\nNote: GPT-OSS-120B requires ~80 GB VRAM for full inference")
    print(f"      With pipeline parallelism across {world_size} GPUs, this should be feasible")

# ============================================================================
# PIPELINE PARALLELISM SETUP
# ============================================================================
# Pipeline parallelism is handled automatically by device_map="auto"
# No manual setup needed - transformers will distribute layers across GPUs

# ============================================================================
# MODEL LOADING
# ============================================================================

if rank == 0:
    print(f"\nLoading {hf_model_name} with Pipeline Parallelism")
    print(f"  Note: This model is very large (~60-65 GB)")
    print(f"  Loading may take several minutes...")
    print(f"  MXFP4 quantization will be used if kernels are available")

# Load tokenizer (all ranks)
try:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    if rank == 0:
        print(f"\nERROR loading tokenizer: {e}")
        print(f"The model path '{hf_model_name}' may not be correct.")
        print(f"Please check HuggingFace for the correct model path.")
        print(f"Alternative paths to try:")
        print(f"  - openai/gpt-oss-120b")
        print(f"  - OpenAI/gpt-oss-120b")
        print(f"  - gpt-oss/gpt-oss-120b")
    dist.destroy_process_group()
    exit(1)

# Load model - strategy depends on size
if rank == 0:
    print(f"  Loading {hf_model_name}...")
    print(f"  Strategy: Pipeline Parallelism with device_map='auto'")
    print(f"  Different layers will be distributed across {world_size} GPUs")

try:
    # Use device_map="auto" for automatic pipeline parallelism
    # This will distribute layers across available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Required for MXFP4 on H100/Hopper GPUs
    )
except Exception as e:
    if rank == 0:
        print(f"\nERROR loading model: {e}")
        print(f"The model path '{hf_model_name}' may not be correct or accessible.")
        print(f"You may need to:")
        print(f"  1. Check the model name is correct")
        print(f"  2. Log in with: huggingface-cli login (if model requires access)")
        print(f"  3. Ensure sufficient RAM for CPU loading")
        if "gpt-oss" in hf_model_name.lower():
            print(f"\n  Note: GPT-OSS-120B requires MXFP4 kernel support!")
            print(f"  Without MXFP4 kernels, it will dequantize to bf16 (~240GB)")
            print(f"  Consider using a smaller model: phi-2, llama-2-7b, etc.")
    dist.destroy_process_group()
    exit(1)

if rank == 0:
    print(f"  ✓ Model loaded with pipeline parallelism")
    print(f"  Layers automatically distributed across GPUs")

# ============================================================================
# PIPELINE PARALLELISM (device_map="auto")
# ============================================================================
# The model is now loaded with layers distributed across GPUs.
# No manual parallelization needed - transformers handles it automatically.

if rank == 0:
    print(f"  Checking model distribution...")

# With device_map="auto", the model is already distributed across GPUs
# Check GPU memory usage to see the distribution
if rank == 0:
    for gpu_id in range(world_size):
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"  GPU {gpu_id}: {allocated:.2f} GB allocated")

# Set to evaluation mode
model.eval()

if rank == 0:
    print(f"\n  ✓ Model initialized with Pipeline Parallelism")
    print(f"  Different layers distributed across {world_size} GPUs")
    print(f"  Communication: Sequential GPU-to-GPU transfers at layer boundaries")
    print(f"  Note: MoE architecture may have additional communication patterns")
    
    # Print final memory usage
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        if allocated > 0:
            print(f"  GPU {i} memory: {allocated:.2f} GB allocated")

# ============================================================================
# INPUT PREPARATION
# ============================================================================

# Prepare inputs
if rank == 0:
    print(f"\nPreparing input sequence:")
    print(f"  Input prompt: '{INPUT_PROMPT[:100]}{'...' if len(INPUT_PROMPT) > 100 else ''}'")

# Tokenize the input prompt
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=False, truncation=False)
input_ids = inputs["input_ids"]

# Get original token count
original_token_count = input_ids.shape[1]

if rank == 0:
    print(f"  Original token count: {original_token_count}")

# Adjust to exact token count if specified
if EXACT_INPUT_TOKENS is not None:
    if original_token_count < EXACT_INPUT_TOKENS:
        # Pad to reach exact length
        pad_token_id = INPUT_PAD_TOKEN_ID if INPUT_PAD_TOKEN_ID is not None else tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        
        num_pad_tokens = EXACT_INPUT_TOKENS - original_token_count
        padding = torch.full((1, num_pad_tokens), pad_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
        
        if rank == 0:
            print(f"  Padded with {num_pad_tokens} tokens (token_id={pad_token_id})")
    
    elif original_token_count > EXACT_INPUT_TOKENS:
        # Truncate to exact length
        input_ids = input_ids[:, :EXACT_INPUT_TOKENS]
        
        if rank == 0:
            print(f"  Truncated by {original_token_count - EXACT_INPUT_TOKENS} tokens")
    
    if rank == 0:
        print(f"  ✓ Adjusted to exactly {EXACT_INPUT_TOKENS} tokens")
        # Show what the adjusted input looks like
        adjusted_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print(f"  Adjusted input: '{adjusted_text[:150]}{'...' if len(adjusted_text) > 150 else ''}'")

# Prepare inputs dictionary and move to device
inputs = {"input_ids": input_ids}
if "attention_mask" in inputs:
    attention_mask = torch.ones_like(input_ids)
    inputs["attention_mask"] = attention_mask

inputs = {k: v.to(f"cuda:{rank}") for k, v in inputs.items()}

# Create output directory (only rank 0)
if rank == 0:
    final_token_count = input_ids.shape[1]
    
    # Clean up old traces to avoid confusion from failed runs
    import shutil
    if os.path.exists(OUTPUT_DIR_BASE):
        print(f"\nCleaning up old GPU traces in {OUTPUT_DIR_BASE}...")
        shutil.rmtree(OUTPUT_DIR_BASE)
    if os.path.exists(CHAKRA_ET_OUTPUT_BASE):
        print(f"Cleaning up old CPU traces in {CHAKRA_ET_OUTPUT_BASE}...")
        shutil.rmtree(CHAKRA_ET_OUTPUT_BASE)
    
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    os.makedirs(CHAKRA_ET_OUTPUT_BASE, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generation Plan")
    print(f"{'='*70}")
    print(f"Input sequence:")
    print(f"  - Token count: {final_token_count} tokens")
    print(f"  - This will be processed in the FIRST forward pass (prefill phase)")
    print(f"\nGeneration phases:")
    if WARMUP_TOKENS > 0:
        print(f"  1. Warmup: Generate {WARMUP_TOKENS} tokens (not profiled)")
        print(f"  2. Profiling: Generate {TOTAL_TOKENS_TO_GENERATE} tokens (each profiled separately)")
        print(f"     - Total forward passes: {TOTAL_TOKENS_TO_GENERATE}")
    else:
        print(f"  1. Profiling: Generate {TOTAL_TOKENS_TO_GENERATE} tokens (each profiled separately)")
        print(f"     - Forward pass 0: Prefill ({final_token_count} tokens) + generate first token")
        print(f"     - Forward passes 1-{TOTAL_TOKENS_TO_GENERATE-1}: Decode (1 new token each)")
        print(f"     - Total forward passes: {TOTAL_TOKENS_TO_GENERATE}")
    
    if PROFILE_EACH_TOKEN:
        print(f"\nTrace generation:")
        print(f"  - Each forward pass will have its own trace file")
        print(f"  - Total trace directories: {TOTAL_TOKENS_TO_GENERATE}")
        print(f"  - Traces per directory: {world_size if PROFILE_ALL_RANKS else 1} (one per rank)")
        print(f"  - Total trace files: {TOTAL_TOKENS_TO_GENERATE * (world_size if PROFILE_ALL_RANKS else 1)}")
    else:
        print(f"  - All tokens in a single trace file")
    
    print(f"\nProfiling configuration:")
    print(f"  - Profile all ranks: {PROFILE_ALL_RANKS}")
    print(f"  - Profile each token separately: {PROFILE_EACH_TOKEN}")
    print(f"  - Output directories:")
    print(f"    • GPU traces (PyTorch): {OUTPUT_DIR_BASE}/")
    print(f"    • CPU traces (Chakra ET): {CHAKRA_ET_OUTPUT_BASE}/")
    print(f"{'='*70}")

# ============================================================================
# PHASE 1: WARMUP GENERATION (NOT PROFILED)
# ============================================================================

# We'll use manual generation loop for precise control over profiling
input_ids = inputs["input_ids"]

if WARMUP_TOKENS > 0:
    if rank == 0:
        print(f"\nPhase 1: Warmup generation ({WARMUP_TOKENS} tokens)...")
        print(f"  This will prepare the model for profiling")
        print(f"  Starting sequence length: {input_ids.shape[1]} tokens")
    
    # Warmup phase - generate tokens without profiling
    with torch.no_grad():
        for warmup_step in range(WARMUP_TOKENS):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if rank == 0 and (warmup_step == 0 or warmup_step == WARMUP_TOKENS - 1):
                current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"  Warmup step {warmup_step + 1}/{WARMUP_TOKENS}: {current_text[-50:]}")  # Show last 50 chars
    
    if rank == 0:
        warmup_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"  ✓ Warmup complete. Current sequence length: {input_ids.shape[1]} tokens")
else:
    if rank == 0:
        print(f"\nPhase 1: Skipping warmup (WARMUP_TOKENS=0)")
        print(f"  Starting directly with profiled generation")
        print(f"  Initial sequence length: {input_ids.shape[1]} tokens")

# ============================================================================
# PHASE 2: PROFILED GENERATION - PER-TOKEN TRACES
# ============================================================================

if rank == 0:
    print(f"\nPhase 2: Profiling {TOTAL_TOKENS_TO_GENERATE} token generations...")
    if PROFILE_EACH_TOKEN:
        print(f"  Each token will have its own trace file")
        print(f"  This may take a while...")
    print()

# Decide whether this rank should profile
should_profile = PROFILE_ALL_RANKS or (rank == 0)

# Store all generated tokens for final output
all_generated_tokens = []

# Profile each token generation separately
for token_idx in range(TOTAL_TOKENS_TO_GENERATE):
    # Determine if this is prefill or decode phase
    is_prefill = (token_idx == 0 and WARMUP_TOKENS == 0)
    phase_name = "PREFILL" if is_prefill else "DECODE"
    
    if rank == 0 and (token_idx % 10 == 0 or token_idx < 5):
        current_seq_len = input_ids.shape[1]
        print(f"  Token {token_idx + 1}/{TOTAL_TOKENS_TO_GENERATE} [{phase_name}]: Input sequence length = {current_seq_len}")
        if token_idx == 0:
            if is_prefill:
                print(f"    → This pass processes {current_seq_len} input tokens + generates 1 new token")
            else:
                print(f"    → This pass processes {current_seq_len} tokens + generates 1 new token")
        if token_idx < 5:
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"    Current text: ...{current_text[-100:]}")  # Show last 100 chars
    
    if should_profile and PROFILE_EACH_TOKEN:
        # Create unique output paths for this token
        token_output_dir = f"{OUTPUT_DIR_BASE}/token_{token_idx:04d}"
        token_chakra_output = f"{CHAKRA_ET_OUTPUT_BASE}/token_{token_idx:04d}_rank{rank}.json"
        
        # Create directory (both ranks need this for GPU traces)
        os.makedirs(token_output_dir, exist_ok=True)
        
        # Synchronize before profiling
        dist.barrier()
        
        if token_idx == 0:
            print(f"[Rank {rank}] Setting up profiling for token {token_idx}")
            print(f"[Rank {rank}] CPU trace output: {token_chakra_output}")
            print(f"[Rank {rank}] GPU trace output: {token_output_dir}/rank{rank}/")
        
        # Setup Chakra ET observer for this token
        et_observer = ExecutionTraceObserver()
        et_observer.register_callback(token_chakra_output)
        et_observer.start()
        
        if token_idx == 0:
            print(f"[Rank {rank}] Chakra ET observer started")
        
        # Setup PyTorch profiler for this token
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{token_output_dir}/rank{rank}"),
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
            with torch.profiler.record_function(f"Token_{token_idx}"):
                with torch.no_grad():
                    # Generate one token
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
            prof.step()
        
        # Stop Chakra ET observer
        et_observer.stop()
        et_observer.unregister_callback()
        
        if token_idx == 0:
            print(f"[Rank {rank}] Chakra ET observer stopped, trace written to {token_chakra_output}")
            import os.path
            if os.path.exists(token_chakra_output):
                file_size = os.path.getsize(token_chakra_output) / 1024 / 1024
                print(f"[Rank {rank}] CPU trace file size: {file_size:.2f} MB")
            else:
                print(f"[Rank {rank}] WARNING: CPU trace file not found!")
        
        # Synchronize after profiling
        dist.barrier()
        
    elif should_profile and not PROFILE_EACH_TOKEN:
        # Profile all tokens together (not implemented in this version, but kept for compatibility)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    else:
        # Non-profiling ranks still participate in generation
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    # Store the generated token (only rank 0 needs this for output)
    if rank == 0:
        all_generated_tokens.append(next_token.item())

final_ids = input_ids

if rank == 0:
    final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    print(f"\n  ✓ Generation complete!")
    print(f"  Final sequence length: {final_ids.shape[1]} tokens")
    print(f"  Final output preview: ...{final_text[-200:]}")  # Show last 200 chars

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

if rank == 0:
    print(f"\n{'='*70}")
    print(f"Profiling complete!")
    print(f"{'='*70}")
    print(f"Model: GPT-OSS-120B (MoE, ~117B params, ~5.1B active)")
    print(f"Strategy: Pipeline Parallelism (device_map='auto')")
    print(f"  - Different transformer layers distributed across {world_size} GPUs")
    print(f"  - Communication: Sequential GPU-to-GPU transfers at layer boundaries")
    print(f"  - Quantization: MXFP4 (4-bit) for efficient memory usage (~60GB total)")
    print(f"  - Note: MoE may have additional sparse communication patterns")
    
    print(f"\nProfiling Summary:")
    print(f"  - Warmup tokens: {WARMUP_TOKENS}")
    print(f"  - Profiled tokens: {TOTAL_TOKENS_TO_GENERATE}")
    print(f"  - Tokens per trace: {'1 (per-token profiling)' if PROFILE_EACH_TOKEN else TOTAL_TOKENS_TO_GENERATE}")
    print(f"  - Total trace files: {TOTAL_TOKENS_TO_GENERATE * (world_size if PROFILE_ALL_RANKS else 1)}")
    
    print(f"\nTrace files organization:")
    if PROFILE_EACH_TOKEN:
        print(f"  GPU traces (PyTorch): {OUTPUT_DIR_BASE}/token_XXXX/rankY/")
        print(f"    - token_0000/ through token_{TOTAL_TOKENS_TO_GENERATE-1:04d}/")
        print(f"    - Each contains traces for {'all ranks' if PROFILE_ALL_RANKS else 'rank 0 only'}")
        print(f"  CPU traces (Chakra ET): {CHAKRA_ET_OUTPUT_BASE}/token_XXXX_rankY.json")
        print(f"    - token_0000_rank0.json through token_{TOTAL_TOKENS_TO_GENERATE-1:04d}_rank{world_size-1 if PROFILE_ALL_RANKS else 0}.json")
    else:
        print(f"  GPU traces: {OUTPUT_DIR_BASE}/rank*/")
        print(f"  CPU traces: {CHAKRA_ET_OUTPUT_BASE}_rank*.json")
    
    print(f"\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        if allocated > 0:
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print(f"\nExpected communication patterns in traces:")
    print(f"  - AllReduce after every attention block (per token)")
    print(f"  - AllReduce after every MLP block (per token)")
    print(f"  - MoE routing and expert computation patterns")
    print(f"  - Most communication-heavy strategy!")
    print(f"  - All {world_size} GPUs computing in parallel")
    
    print(f"\nAnalysis tips:")
    print(f"  - Compare traces across tokens to see pattern evolution")
    print(f"  - Early tokens (prefill phase) may have different patterns")
    print(f"  - Later tokens (decode phase) should show consistent patterns")
    print(f"  - Use TensorBoard to visualize: tensorboard --logdir={OUTPUT_DIR_BASE}")
    print(f"{'='*70}")

# Cleanup distributed
dist.destroy_process_group()

