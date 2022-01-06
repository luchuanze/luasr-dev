import torch


def subsequnet_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu")
) -> torch.Tensor:

    ret = torch.zeros(size, size, device=device, dtype=torch.bool)

    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        end = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:end] = True

    return ret


ten = subsequnet_chunk_mask(10, 2, 2)

print(ten)