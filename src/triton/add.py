import torch
import triton
import triton.language as tl

print(triton.__version__)
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(out_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and out.device == DEVICE
    n_elements = out.numel()
    # print(n_elements)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out

def main():
    torch.manual_seed(1337)
    size = 65536
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    out_torch = x + y
    out_triton = add(x, y)
    print(f"torch output\t=\t{out_torch}")
    print(f"triton output\t=\t{out_triton}")
    print(f"max diff\t=\t{torch.max(torch.abs(out_torch - out_triton))}")


if __name__ == "__main__":
    main()