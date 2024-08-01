import torch
from axonn import axonn as ax
from axonn.intra_layer import Linear, enable_timers, timers
from axonn.intra_layer import optimize_communication, clear_weights_cache
import os
import numpy as np
import json
from checkpoint_activations import checkpoint

def parse_flops(input_string):
    # Use a regular expression to find the pattern
    import re
    match = re.search(r'(\d+),(\d+),(\d+)', input_string)

    if match:
        # Extract the numbers and convert them to integers
        num1, num2, num3 = map(int, match.groups())
        return num1*num2*num3*2/1e12
    else:
        return None

def log_timers(skip=False):
    compute_times, events = timers.get_times()
    if skip:
        return
    keys_for_compute_times = list(compute_times.keys())
    keys_for_compute_times.sort()
    all_tflops = []
    for k in keys_for_compute_times:
        this_compute_time = compute_times[k]
        this_num = events[k]
        msg = f"Compute Time - {k} | "
        tflops = parse_flops(k) * this_num 
        msg += f" | FLOP/s = {tflops/this_compute_time:.3f} TFLOP/s | % of peak {tflops*100/this_compute_time/192:.3f}"
        if torch.distributed.get_rank() == 0:
            print(msg)
        all_tflops.append(tflops*100/this_compute_time/192)
    return all_tflops


class MLP(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_proj = Linear(H, 4*H, bias=False)
        self.out_proj = Linear(4*H, H, transpose=True, bias=False)
        self.act = torch.nn.GELU()

    def _forward(self, x):
        h = self.in_proj(x, scatter_input=False, gather_output=False)
        h = self.act(h)
        h = self.out_proj(h, scatter_input=False, gather_output=False)
        return x + h

    def forward(self, x):
        return checkpoint(self._forward, self, x)

def benchmark(B, S, H, N, iters=10, overlap=False):
    model = torch.nn.Sequential(*[MLP(H) for _ in range(N)]).cuda()
    x = torch.randn(B, S, H//ax.config.G_intra_c, device="cuda", dtype=torch.bfloat16)
    x.requires_grad = True
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with optimize_communication(overlap, overlap, overlap, model):
                y = model(x)
                y.backward(y.detach())
            clear_weights_cache()
        all_tflops = log_timers()
        if torch.distributed.get_rank() == 0:
            print("===================")
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000
    time_per_iter = total_time / iters

    return all_tflops, time_per_iter


if __name__ == "__main__":
    B = 1
    S = 2048
    N = 5
    torch.distributed.init_process_group(backend="nccl")
    G_depth = torch.distributed.get_world_size() // 2
    ax.init(G_inter=1, G_data=1, G_intra_r=2, G_intra_c=1, 
            G_intra_d = G_depth)
    enable_timers()
    results = {}
    for H in [16384]:
        tflops, time_per_iter = benchmark(B, S, H, N=N, overlap=False)
        tflops = np.sort(tflops)
        results[H] = list(tflops)
        tflops_per_gpu = 64 * (B * G_depth) * S * N * H * H / 1e12 / time_per_iter / torch.distributed.get_world_size()
        if torch.distributed.get_rank() == 0:
            print(f"B={B} S={S} H={H} N={N} | TFLOP/s={tflops_per_gpu:.2f} TFLOP/s")

    if torch.distributed.get_rank() == 0:
        print(json.dumps(results))
     
