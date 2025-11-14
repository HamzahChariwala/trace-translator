"""
Profile a small LLM using PyTorch Profiler for Chakra trace collection.
"""

import torch
import torch.profiler
from model_loader import ModelLoader

# ============================================================================
# CONFIGURATION - Adjust these to control trace size
# ============================================================================

MODEL_NAME = "tinyllama"           # Which model to profile
DEVICE = "cpu"                     # Device: 'cpu', 'cuda', or 'mps'
INPUT_PROMPT = "The quick brown fox jumps over the lazy dog."  # Input text

# Profiling schedule parameters (control trace size)
WAIT_STEPS = 1                     # Skip first N iterations (not profiled)
WARMUP_STEPS = 0                   # Warmup iterations (profiled but marked as warmup)
ACTIVE_STEPS = 1                   # Number of iterations to actively profile
TOTAL_ITERATIONS = 2               # Total forward passes (must be >= wait + warmup + active)

OUTPUT_DIR = './traces'            # Where to save trace files

# ============================================================================

# Load model
loader = ModelLoader(device=DEVICE)
model, tokenizer = loader.load_reasoning_model(MODEL_NAME)

# Prepare inputs
inputs = tokenizer(INPUT_PROMPT, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# Note: Token generation is controlled by the input prompt length
# The model processes all tokens in the prompt in one forward pass
# To profile generation (producing new tokens), use model.generate() instead of model(**inputs)

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(
        wait=WAIT_STEPS,
        warmup=WARMUP_STEPS,
        active=ACTIVE_STEPS,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(OUTPUT_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(TOTAL_ITERATIONS):
        with torch.no_grad():
            model(**inputs)
        prof.step()

print(f"Trace saved to {OUTPUT_DIR}/")
