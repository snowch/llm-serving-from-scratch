---
title: "GPU and Hardware Primer for Serving"
short_title: "Appendix A"
---

(appendix-a)=
# Appendix A · GPU and Hardware Primer for Serving [DRAFT]

[To write: GPU and Hardware Primer for Serving.]

## Planned contents

- HBM vs SRAM vs registers: the memory hierarchy that makes FlashAttention (ch12) make sense
- SMs, warps and occupancy — only as deep as serving decisions require
- Tensor cores and supported dtypes, and what that means for ch14's format choices
- NVLink vs PCIe, and why the interconnect decides whether ch11 and ch17 are viable
- MIG and GPU partitioning as an alternative to multi-tenancy (ch19)
- **The spec-sheet numbers that actually predict serving performance**: HBM bandwidth, HBM capacity, and little else
