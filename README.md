# Open R1

## Overview

This repository provides a reproducible workflow for training and evaluating Hugging Face’s [open-r1](https://github.com/huggingface/open-r1) model in a high-performance computing (HPC) environment that differs from the one used by the original developers.

While the original `open-r1` repository assumes access to a node with 8×H100 GPUs, many HPC setups offer nodes with only 4 GPUs. This guide demonstrates how to adapt the training process to nodes with 4 GPUs. In particular, the provided configuration supports training across 2 nodes, each equipped with 4×H100 GPUs.

Since Open R1 is an actively developed project, the upstream repository changes frequently and the official README may not always reflect the latest updates. This branch targets a specific commit of the original repository, and the steps provided here are tailored accordingly to ensure reproducibility.

## Installation

The original repository recommends using [`uv`](https://github.com/astral-sh/uv) to manage dependencies. However, `uv` can be directly installed in HPC environments.

To work around this:
1. First, create a Python virtual environment:
   ```bash
   python -m venv openr1
   source openr1/bin/activate   
2. Then install `uv` inside this virtual environment:
   ```bash
   pip install uv
3. Once `uv` is installed, follow the installation steps from the original repository to install all required dependencies.

## Training models

> [!NOTE]  
> The training setup below is configured for **2 nodes**, each equipped with **4×H100 (80GB)** GPUs.  
> For different hardware configurations or topologies, adjust the per-device batch size and the number of gradient accumulation steps to maintain the same **global batch size**.

> **Global Batch Size** = (Per-Device Batch Size × Number of GPUs) × Gradient Accumulation Steps

To perform supervised fine-tuning (SFT) using DeepSpeed ZeRO-3 on 2 nodes, run the following command:

```bash
sbatch slurm/quick_sfttrain.slurm
```
This script fine-tunes the `Qwen2.5-Math-7B-Instruct` model on the `OpenR1-Math-220k` dataset using configurations optimized for multi-node training.OpenR1-Math-220k dataset using configurations optimized for multi-node training.

## Evaluating models

Once training is complete, model evaluation is performed using [`lighteval`](https://github.com/huggingface/lighteval).

To run evaluation across 2 nodes, use the following Slurm command:

```bash
sbatch slurm/quick_evaluate.slurm
```

This script evaluates the fine-tuned model on the `MATH-500` benchmark dataset. The results will be saved in the `logs` directory, typically in the `slurm.out` file.

## Results
<div align="center">

<table>
  <thead>
    <tr>
      <th style="text-align:center;"><b>Model</b></th>
      <th style="text-align:center;"><b>AIME 2024</b> pass@1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">
        <b>Qwen2.5-Math-7B-Instruct</b><br>
        <sub>(Original model)</sub>
      </td>
      <td style="text-align:center;">10.0</td>
    </tr>
    <tr>
      <td style="text-align:center;">
        <b>Qwen2.5-Math-7B-Instruct (Fine-tuned)</b><br>
        <sub>(Supervised Fine-tuned on data distilled from DeepSeek R1)</sub>
      </td>
      <td style="text-align:center;">66.7</td>
    </tr>
    <tr>
      <td style="text-align:center;">
        <b>DeepSeek-R1-Distill-Qwen-7B</b><br>
        <sub>(Reportedly fine-tuned on its own distilled data)</sub>
      </td>
      <td style="text-align:center;">50.0</td>
    </tr>
  </tbody>
</table>

</div>

## Acknowledgements

I would like to thank all the contributors to the [open-r1](https://github.com/huggingface/open-r1) project for their valuable work. I am also grateful to the Center for Information Services and High Performance Computing — [Zentrum für Informationsdienste und Hochleistungsrechnen (ZIH)] at TU Dresden — for providing access to its facilities for high-throughput computations.
