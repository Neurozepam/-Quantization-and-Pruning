import time
import torch

def measure_cuda_call(fn):

    """Measures the time taken for a CUDA call to complete."""
    
    torch.cuda.synchronize()  
    start_time = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return result, elapsed_seconds


def benchmark_cuda(fn):

    """Benchmarks a CUDA function by measuring its execution time and memory usage."""

    measure_cuda_call(fn)

    torch.cuda.reset_peak_memory_stats()

    durations = []
    for _ in range(3):
        _, elapsed_seconds = measure_cuda_call(fn)
        durations.append(elapsed_seconds)

    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    
    return {
        "seconds": durations,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }