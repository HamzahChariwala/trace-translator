# PyTorch Trace Collection for Chakra

## What Chakra Expects

Chakra's PyTorch converter expects **Kineto JSON traces** with:
- Tensor shapes (`record_shapes=True`)
- Memory usage (`profile_memory=True`)
- Call stack (`with_stack=True`)
- CPU and optionally CUDA activities

## Collection Code

### Custom Model

```python
import torch
import torch.nn as nn
import torch.profiler

# Your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MyModel().cuda()
inputs = torch.randn(32, 512).cuda()

# Profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./traces'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        model(inputs)
        prof.step()
```

### Open-Source Model (Hugging Face)

```python
import torch
import torch.profiler
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("bert-base-uncased").cuda()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(["Sample text"], return_tensors="pt", padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}

# Profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./traces'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        model(**inputs)
        prof.step()
```

## Output

**Generated file**: `./traces/<timestamp>.pt.trace.json`

This is a Kineto-format JSON containing CPU/GPU operators, tensor metadata, and timing information.

## CPU-Only (No CUDA)

Remove `ProfilerActivity.CUDA` from activities list. Everything else stays the same.

