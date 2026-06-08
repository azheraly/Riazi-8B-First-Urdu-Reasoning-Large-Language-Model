# Urdu Reasoning LLM

## Aqal: The First Urdu Reasoning Large Language Model through Continued Pretraining and GRPO-Based Reinforcement Learning

------------------------------------------------------------------------

# 1. Project Overview

Aqal is the first reasoning-optimized Urdu Large Language Model (LLM) developed to improve multi-step reasoning, logical consistency, and final-answer correctness in Urdu. This project leverages a three-stage training pipeline combining Continued Pretraining, Supervised Fine-Tuning, and GRPO-based Reinforcement Learning.

1.  Continued Pretraining (CPT)\
2.  Supervised Fine-Tuning (SFT)\
3.  GRPO-based Reinforcement Learning (RL)

The system focuses on improving multi-step reasoning, logical
consistency, and final-answer correctness in Urdu.

------------------------------------------------------------------------

# 3. Environment Setup

## Using Conda (Recommended)

conda create --name venv\
conda activate venv

## Using pip

pip install -r requirements.txt

Python 3.10+ recommended.

------------------------------------------------------------------------

# 4. Training Pipeline

## Continued Pretraining and  Supervised Fine-Tuning

python script.py


## GRPO Reinforcement Learning

python training/grpo_trainer.py

------------------------------------------------------------------------

# 5. Inference

python evaluation/inference.py 

------------------------------------------------------------------------

