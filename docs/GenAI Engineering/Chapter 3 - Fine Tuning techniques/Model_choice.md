# How to choose the right model

## Main considerations

Model Size VRAM (FP16) VRAM (4-bit) Cloud Options Local Hardware Best Use Cases
1–3B 4–6 GB ~2 GB AWS g4dn.xlarge, basic GPU instances RTX 3060, laptop GPUs Basic chat, text classification, autocomplete
7–8B 14–16 GB ~6–8 GB AWS g5.xlarge, RunPod RTX 4090 RTX 4080/4090, A6000 General-purpose assistants, summarization, coding
13–14B 26–28 GB ~12–16 GB AWS g5.2xlarge, multi-instance RTX 4090 (quantized only) Stronger reasoning, better instruction following
70B+ 140 GB+ ~35–40 GB AWS p4d.24xlarge, A100 clusters Multi-GPU setups (expensive) SOTA reasoning, enterprise applications
