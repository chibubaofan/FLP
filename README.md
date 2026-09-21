# FLP: Foresight-Guided Defense against Infectious Jailbreak in MAS

<p align="center">
  <i>Catching the Infection Before It Spreads</i>
</p>
<p align="center">
  <a href="https://arxiv.org/pdf/2605.01758">📄 Paper</a> 
</p>

---

## Overview

Multi-agent systems built on large multimodal models are vulnerable to infectious jailbreak attacks. Prior defenses introduce a global healing factor to suppress the effect of adversarial samples, but this only achieves superficial suppression rather than truly restoring infected agents. We recognize a fundamental mismatch: global healing factors operate system-wide, while viral adversarial samples act locally. To address this, we propose **FLP**, which uses multi-persona simulation to obtain rich observable signals, performs infection diagnosis via joint semantic diversity and retrieval entropy, and achieves true recovery of infected agents through localized, precise identification and removal.

---

## Project Structure

```
FLP/
├── simultaion/                  # Multi-agent simulation and FLP defense
│   ├── simulation_FLP.py        # Main simulation script (FLP defense)
│   ├── simulation_batch.py      # Batch benign simulation
│   ├── simulation_test_batch.py # AgentSmith attack simulation
│   ├── detection_performance.py # Detection performance evaluation
│   ├── get_detection_tao.py     # Detection threshold calibration
│   ├── prompt.py                # Agent prompts and state definitions
│   ├── simultaion_FLP.sh      
│   ├── simulation_attack.sh   
│   ├── simulation_benign.sh   
│   ├── detection_performance.sh 
│   └── get_detection_tao.sh   
├── train_adv_image/             # Adversarial image generation
│   ├── optimize.py              # Border / pixel adversarial attack optimization
│   ├── validate.py              # Adversarial image validation
│   └── train.sh                 # One-click adversarial image training
├── data/                        # Data files
│   ├── million_villagers_1024.json       # Agent persona data
│   └── million_villagers_1024_test.json
│   ├── album_pool/              # Images used during interaction
│   └── attack_image/            # Provided attack images
├── analyze/
│   └── analyze.ipynb            # Result analysis and visualization
└── requirements.txt
```

---

## Requirements

Please install dependencies from `requirements.txt`.

---

## Quick Start

Minimal example: run the FLP defense simulation.

```bash
cd simultaion
bash simultaion_FLP.sh
```

## Key Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--num_agents` | 128 | Number of agents during simulation |
| `--num_rounds` | 65 | Total number of interaction rounds |
| `--num_attacks` | 4 | Number of attacking agents (`0` for benign simulation) |
| `--malicious_threshold` | 0 | Infection diagnosis threshold per agent |
| `--slice_size` | 2 | Minimum memory slice size for healing diagnosis |
| `--vlm` | llava-hf/llava-1.5-7b-hf | VLM model path |
| `--clip` | openai/clip-vit-large-patch14 | CLIP model path |
| `--batch_size` | 8 | Inference batch size |
| `--seed` | 42 | Random seed |

---

## Acknowledgments

Thanks to the open-source project AgentSmith: [sail-sg/Agent-Smith: [ICML 2024] Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://github.com/sail-sg/Agent-Smith)
