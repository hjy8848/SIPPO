"""Run a minimal FlashAttention CUDA forward/backward compatibility check."""

import torch
from flash_attn import flash_attn_func


def main() -> None:
    query = torch.randn(
        2,
        256,
        8,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    output = flash_attn_func(query, key, value, causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    if not output.isfinite().all():
        raise RuntimeError("FlashAttention forward produced non-finite values")
    if query.grad is None or not query.grad.isfinite().all():
        raise RuntimeError("FlashAttention backward produced invalid gradients")
    print(
        "FLASH_ATTN_FWD_BWD_OK",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
        tuple(output.shape),
    )


if __name__ == "__main__":
    main()
