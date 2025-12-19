<div align="center">

# Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images 

<!-- ### Arxiv 2025 -->

<p align="center">  
    <a href="https://jkhu29.github.io/">JiaKui Hu</a>,
    <a href="https://sshan-zhao.github.io/">Shanshan Zhao♯</a>,
    <a href="https://scholar.google.com/citations?user=GlqRHLcAAAAJ">Qing-Guo Chen</a>,
    <a href="https://bollossom.github.io/">Xuerui Qiu</a>,
    <a href="https://scholar.google.com/citations?user=OkMMP2AAAAAJ">Jialun Liu</a>,
    <a href="https://scholar.google.com/citations?user=uiDNWw0AAAAJ">Zhao Xu</a>,
    <a href="https://scholar.google.com/citations?user=tsKl9GUAAAAJ">Weihua Luo</a>,
    <a href="https://openreview.net/profile?id=~Kaifu_Zhang2">Kaifu Zhang</a>,
    <a href="https://scholar.google.com/citations?user=WSFToOMAAAAJ">Yanye Lu♯</a>
</p>

<p>Peking University, Alibaba International Digital Commerce Group</p>

</div>

<div align="center">
    <a href="https://jkhu29.github.io/omni_view/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2511.07222"><strong>Paper</strong></a> |
  <a href="https://huggingface.co/AIDC-AI/Omni-View/tree/main"><strong>Model</strong></a> 
</div>

<br>

<!-- > [JiaKui Hu](https://jkhu29.github.io/), [Shanshan Zhao♯](https://sshan-zhao.github.io/), [Qing-Guo Chen](https://scholar.google.com/citations?user=GlqRHLcAAAAJ&hl=en), [Xuerui Qiu](https://bollossom.github.io/), [Jialun Liu](https://scholar.google.com/citations?user=OkMMP2AAAAAJ), [Zhao Xu](https://scholar.google.com/citations?user=uiDNWw0AAAAJ), [Weihua Luo](https://scholar.google.com/citations?user=tsKl9GUAAAAJ), [Kaifu Zhang](https://openreview.net/profile?id=~Kaifu_Zhang2), [Yanye Lu♯](https://scholar.google.com/citations?user=WSFToOMAAAAJ)
>
> contact: jkhu29@stu.pku.edu.cn
>  -->
> This paper presents **Omni-View**, which extends the unified multimodal understanding and generation to 3D scenes based on multiview images, exploring the principle that "generation facilitates understanding". Consisting of understanding model, texture module, and geometry module, Omni-View jointly models scene understanding, novel view synthesis, and geometry estimation, enabling synergistic interaction between 3D scene understanding and generation tasks. By design, it leverages the spatiotemporal modeling capabilities of its texture module responsible for appearance synthesis, alongside the explicit geometric constraints provided by its dedicated geometry module, thereby enriching the model’s holistic understanding of 3D scenes. Trained with a two-stage strategy, Omni-View achieves a state-of-the-art score of 55.4 on the VSI-Bench benchmark, outperforming existing specialized 3D understanding models, while simultaneously delivering strong performance in both novel view synthesis and 3D scene generation.

<p align="center"><img src="assets/teaser.png" width="95%"></p>


## 🔥 Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/AIDC-AI/Omni-View.git
cd Omni-View
conda create -n omniview python=3.10 -y
conda activate omniview
pip install torch==2.6.0 torchvision # please following https://pytorch.org/get-started/previous-versions/
pip install -r requirements.txt
pip install flash_attn==2.7.4 --no-build-isolation
```

2️⃣  Download the pre-trained checkpoint of BAGEL and Omni-View
```python
# BAGEL, configs and VAE
from huggingface_hub import snapshot_download

save_dir = "./pretrained_model/BAGEL-7B-MoT/"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "ae.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

```shell
# Omni-View
huggingface-cli download AIDC-AI/Omni-View --local-dir ./
```

## 🔥 Eval

### Eval
We provide the scripts for evaluating 3D scene understanding, Spatial Reasoning (VSI-bench), and Novel View Synthesis. 
Please See [EVAL](EVAL.md) for more details.

## 📊 Benchmarks

### 1. 3D Scene Understanding

<p align="center"><img src="assets/3d_und.png" width="95%"></p>

### 2. VSI-Bench

<p align="center"><img src="assets/vsi.png" width="95%"></p>

### 3. Novel View Synthesis

<p align="center"><img src="assets/nvs.png" width="95%"></p>


## ✍️ Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{hu2025omniview,
      title={Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images}, 
      author={JiaKui Hu and Shanshan Zhao and Qing-Guo Chen and Xuerui Qiu and Jialun Liu and Zhao Xu and Weihua Luo and Kaifu Zhang and Yanye Lu},
      year={2025},
      eprint={2511.07222},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.07222}, 
}
```

## 🧡 Acknowledgements

Our implementation is built upon [Bagel](https://github.com/bytedance-seed/BAGEL). We appreciate their great work.


## 📄 License

```
Copyright (C) 2025 AIDC-AI
Licensed under the Apache License, Version 2.0.
This project contains various third-party components under other open source licenses. You should respect the terms of those licenses.
The component DiT is released under the CC-BY-NC 4.0 License (for non-commercial purposes only).
See the NOTICE file for more information.
```
