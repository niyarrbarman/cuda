import torch
import time
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
torch.manual_seed(1337)


def naive_softmax(x) -> torch.Tensor:
    """

    x = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    
    x[0] = [1, 2, 3, 4, 5]
    x[0] = [-4, -3, -2, -1, 0]

    numerator = torch.exp(x[0]) --> [torch.exp(-4), torch.exp(-3), ...]
    denominator = torch.sum(numerator)

    softmax = numerator / denominator
    """    
    x_max = x.max(dim=1)[0]
    x = x - x_max[:, None]
    numerator = torch.exp(x)
    denominator = torch.sum(numerator, dim=1)
    output = numerator / denominator[:, None]
    return output


@triton.jit
def _softmax_kernel(
    out_ptr,
    stride_out,
    x_ptr,
    stride_x,
    cols,
    block_size: tl.constexpr,
    num_warps: tl.constexpr,
):
    
    row_idx = tl.program_id(0)

    row_start_ptr = x_ptr + (row_idx * stride_x)
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < cols

    x = tl.load(input_ptrs, mask=mask, other=float('-inf'))

    x_max = tl.max(x, axis=0)
    safe_x = x - x_max

    numerator = tl.exp(safe_x)
    denominator = tl.sum(numerator, axis=0)

    softmax = numerator/denominator

    tl.store(out_ptr + col_offsets + row_idx * stride_out, softmax, mask=mask)

    

def triton_softmax(
        x: torch.Tensor,
        ) -> torch.Tensor:
    rows, cols = x.shape
    assert len(x.shape) == 2, f"only 2d matrices for now"
    out = torch.empty_like(x)
    assert x.device == DEVICE and out.device == DEVICE, f"device not cuda"
    block_size = triton.next_power_of_2(cols)
    num_warps = min(4, triton.cdiv(block_size, 32))
    grid = (rows,)

    _softmax_kernel[grid](
        out,
        out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size,
        num_warps,
    )

    return out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 8192*2},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)


