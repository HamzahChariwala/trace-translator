"""
Profile a small LLM using PyTorch Profiler with Model Parallelism across multiple GPUs.
Uses HuggingFace's device_map="auto" to automatically distribute layers across GPUs.
"""

import torch
import torch.profiler
from torch.profiler import ExecutionTraceObserver
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "llama-2-13b"         # Options: 'phi-2', 'tinyllama', 'phi-3-mini', 'llama-2-13b', 'mistral-7b', or HF model path
INPUT_PROMPT = "Tell me about the corpus callosum."

MAX_MEMORY_PER_GPU = {0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"}

# Generation parameters
WARMUP_TOKENS = 5              # Generate this many tokens before profiling
PROFILED_TOKENS = 1             # Profile this many token generations
COOLDOWN_TOKENS = 10           # Generate this many more after profiling (to see response)

OUTPUT_DIR = './traces_parallel'
CHAKRA_ET_OUTPUT = './CPU_trace_parallel'

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

# Check GPUs
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Model parallelism requires multiple GPUs.")

num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPU(s)")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Load model with device_map="auto" for automatic multi-GPU distribution
print(f"\nLoading {hf_model_name} with device_map='auto'")
if MAX_MEMORY_PER_GPU:
    print(f"Memory constraints: {MAX_MEMORY_PER_GPU}")

tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Llama models need this

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_safetensors=True,
    device_map="auto",  # Automatically distributes across GPUs
    max_memory=MAX_MEMORY_PER_GPU,  # Optional memory constraints per device
    low_cpu_mem_usage=True
)
model.eval()

# Show device placement
if hasattr(model, 'hf_device_map'):
    print("\nDevice placement:")
    for name, device in model.hf_device_map.items():
        print(f"  {name}: {device}")

# Prepare inputs and move to first device
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=True, truncation=True)
first_device = "cuda:0"
if hasattr(model, 'hf_device_map'):
    for device in model.hf_device_map.values():
        if isinstance(device, int):
            first_device = f"cuda:{device}"
            break
inputs = {k: v.to(first_device) for k, v in inputs.items()}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nGeneration plan:")
print(f"  1. Warmup: Generate {WARMUP_TOKENS} tokens (not profiled)")
print(f"  2. Profile: Generate {PROFILED_TOKENS} token(s) (PROFILED)")
print(f"  3. Cooldown: Generate {COOLDOWN_TOKENS} more tokens (not profiled)")
print(f"  Total tokens: {WARMUP_TOKENS + PROFILED_TOKENS + COOLDOWN_TOKENS}")

# Phase 1: Warmup generation (not profiled)
print(f"\nPhase 1: Warmup generation ({WARMUP_TOKENS} tokens)...")
with torch.no_grad():
    warmup_ids = model.generate(
        **inputs,
        max_new_tokens=WARMUP_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
warmup_text = tokenizer.decode(warmup_ids[0], skip_special_tokens=True)
print(f"  After warmup: {warmup_text}")

# Phase 2: Profiled generation
print(f"\nPhase 2: Profiling {PROFILED_TOKENS} token(s)...")
et_observer = ExecutionTraceObserver()
et_observer.register_callback(CHAKRA_ET_OUTPUT + ".json")
et_observer.start()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler(OUTPUT_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        profiled_ids = model.generate(
            input_ids=warmup_ids,  # Continue from warmup
            max_new_tokens=PROFILED_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

et_observer.stop()
output_file = et_observer.get_output_file_path()
et_observer.unregister_callback()

profiled_text = tokenizer.decode(profiled_ids[0], skip_special_tokens=True)
print(f"  After profiling: {profiled_text}")

# Phase 3: Cooldown generation (not profiled)
print(f"\nPhase 3: Cooldown generation ({COOLDOWN_TOKENS} tokens)...")
with torch.no_grad():
    final_ids = model.generate(
        input_ids=profiled_ids,  # Continue from profiled
        max_new_tokens=COOLDOWN_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
print(f"  Final output: {final_text}")

print(f"\n{'='*60}")
print(f"Profiling complete!")
print(f"{'='*60}")
print(f"Kineto trace saved to: {OUTPUT_DIR}/")
print(f"Chakra ET trace saved to: {output_file}")
print(f"\nGPU Memory Usage:")
for i in range(num_gpus):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    if allocated > 0:
        print(f"  GPU {i}: {allocated:.2f} GB")
print(f"{'='*60}")

