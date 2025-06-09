import torch
from nets.net import Net


net = Net()

torch.cuda.empty_cache()
net.cuda().train()
x = torch.randn(35_000, 3, 32, 32, device="cuda")

for use_amp in (False, True):
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = net(x)
        loss = out.sum()
        loss.backward()  # <-- peak happens inside here
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak allocated ({'AMP' if use_amp else 'FP32'}): {peak:.2f} GB")
