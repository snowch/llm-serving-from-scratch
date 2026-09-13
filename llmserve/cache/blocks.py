"""Paged KV cache: fixed-size blocks and a block table, after vLLM's PagedAttention.

Chapter 7's engine gives every sequence its own contiguous cache tensor. That is simple and it
wastes memory in two ways at once, both of which this module removes.

**Internal fragmentation.** A sequence's final length is unknown when it starts, so a contiguous
allocator must reserve for the longest it might become. A request that reserves 2048 slots and
generates 40 leaves 2008 unusable — not free, *reserved*.

**External fragmentation.** Freed regions of different sizes leave holes too small to hold the
next sequence, so memory can be available in total and unusable in practice.

The fix is the same one operating systems reached decades ago: stop allocating contiguous ranges.
Split memory into fixed-size blocks, give each sequence a table mapping its logical positions to
physical blocks, and allocate a block at a time on demand. A sequence then wastes at most one
partly filled block, and any free block fits any sequence.
"""

from __future__ import annotations

import torch


class OutOfBlocksError(RuntimeError):
    """Raised when no free block is available.

    This is not a bug but a scheduling event: the engine responds by preempting a sequence
    (chapter 8) rather than by crashing.
    """


class BlockAllocator:
    """A pool of fixed-size KV blocks with reference counting.

    Reference counts exist for chapter 9: when two sequences share a prompt prefix they share the
    blocks holding it, and a block may only be freed once every sequence using it is done. Until
    then every count is 1 and the mechanism is invisible.
    """

    def __init__(self, n_blocks: int, block_size: int) -> None:
        if n_blocks < 1 or block_size < 1:
            raise ValueError("n_blocks and block_size must both be positive")
        self.n_blocks = n_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(n_blocks))
        self._refcount: dict[int, int] = {}

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_used(self) -> int:
        return self.n_blocks - len(self._free)

    @property
    def utilisation(self) -> float:
        return self.n_used / self.n_blocks

    def blocks_needed(self, n_tokens: int) -> int:
        """How many blocks a sequence of this length occupies, rounded up."""
        return (n_tokens + self.block_size - 1) // self.block_size

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocksError(f"requested {n} blocks, {len(self._free)} free")
        taken = [self._free.pop() for _ in range(n)]
        for block in taken:
            self._refcount[block] = 1
        return taken

    def refcount(self, block: int) -> int:
        """How many holders a block has. Zero means it is in the free pool."""
        return self._refcount.get(block, 0)

    def share(self, block: int) -> int:
        """Add a reference to an existing block (chapter 9's prefix sharing)."""
        self._refcount[block] += 1
        return block

    def free(self, blocks: list[int]) -> None:
        """Drop a reference to each block, returning those that reach zero to the pool."""
        for block in blocks:
            count = self._refcount.get(block, 0)
            if count <= 1:
                self._refcount.pop(block, None)
                self._free.append(block)
            else:
                self._refcount[block] = count - 1


class PagedKVCache:
    """Block-structured KV storage for every layer at once.

    Keys and values live in one tensor per layer, indexed by physical block rather than by
    sequence. A sequence is then just a list of block numbers, which is what makes both the
    fragmentation fix and chapter 9's sharing possible.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.allocator = BlockAllocator(n_blocks, block_size)
        self.block_size = block_size
        self.n_layers = n_layers
        shape = (n_blocks, n_kv_heads, block_size, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype) for _ in range(n_layers)]
        self.values = [torch.zeros(shape, dtype=dtype) for _ in range(n_layers)]

    def total_bytes(self) -> int:
        per_tensor = self.keys[0].numel() * self.keys[0].element_size()
        return 2 * self.n_layers * per_tensor

    def write(
        self, layer: int, block_table: list[int], start: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write ``k``/``v`` for tokens at logical positions ``start ...`` into their blocks.

        ``k`` and ``v`` are [n_kv_heads, n_tokens, head_dim]. Tokens are placed one at a time
        because a run may straddle a block boundary; a real kernel does this with a scatter, which
        is what chapter 13 writes.
        """
        for i in range(k.shape[1]):
            position = start + i
            block = block_table[position // self.block_size]
            offset = position % self.block_size
            self.keys[layer][block, :, offset, :] = k[:, i, :]
            self.values[layer][block, :, offset, :] = v[:, i, :]

    def gather(
        self, layer: int, block_table: list[int], length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise a sequence's cache as contiguous [1, n_kv_heads, length, head_dim].

        Deliberately the simplest thing that works, and deliberately wasteful: it copies the whole
        cache on every step. Chapter 13 replaces it with a kernel that reads the blocks in place.
        """
        n_blocks = self.allocator.blocks_needed(length)
        used = block_table[:n_blocks]
        n_heads, _, head_dim = self.keys[layer].shape[1:]
        # [blocks, heads, block_size, dim] -> [heads, blocks * block_size, dim]
        keys = self.keys[layer][used].permute(1, 0, 2, 3).reshape(n_heads, -1, head_dim)
        values = self.values[layer][used].permute(1, 0, 2, 3).reshape(n_heads, -1, head_dim)
        return keys[None, :, :length, :], values[None, :, :length, :]
