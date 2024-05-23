import torch
import os
from datetime import datetime
from functools import partial
def trace_handler(
    prof: torch.profiler.profile,
    rank: int,
    export_trace=True,
    export_memory_timeline=False,
    with_stack: bool = True,
    group_by_stack: int = 0,
    group_by_input_shapes: bool = False,
    prefix="",
    out_dir="./profiles",
    time_fmt_str: str = "%m_%d_%H",
    metric="self_cuda_time_total",
    row_limit=25,
    verbose=False,
):
    # Prefix for file names.
    timestamp = datetime.now().strftime(time_fmt_str)
    file_prefix = os.path.join(out_dir, f"{prefix}-{timestamp}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Construct the trace file.
    if export_trace:
        prof.export_chrome_trace(f"{file_prefix}-chrome-trace.json.gz")

    # Construct the memory timeline file.
    if export_memory_timeline:
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.html"
        )

    if with_stack:
        prof.export_stacks(f"{file_prefix}-stacks.txt", metric=metric)
    
    key_avgs = prof.key_averages(
        group_by_input_shape=group_by_input_shapes, group_by_stack_n=group_by_stack
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{file_prefix}-key_averages.txt", "w") as f:
        print(
            key_avgs, file=f
        )
    if rank == 0:
        print(f"Saving profiling results to {out_dir}")
        if verbose:
            print(key_avgs)
            
class FakeContext:
    """
    Fake context when not using profiler with profiling script.
    
    """
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        pass
    
    def step(self):
        pass