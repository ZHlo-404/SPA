# 🚀 Structure-aware Prompt Adaptation (SPA)
> Official PyTorch implementation of **"Structure-aware Prompt Adaptation from Seen to Unseen for Open-Vocabulary Compositional Zero-Shot Learning" 

## ✨ Overview
**SPA (Structure-aware Prompt Adaptation)** is a **plug-and-play module** that improves **Open-Vocabulary Compositional Zero-Shot Learning (OV-CZSL)** by preserving and leveraging the **local semantic structure** in CLIP’s embedding space.
SPA introduces two complementary components:
- 🧩 **Structure-aware Consistency Loss (SCL)** — preserves the local topology among *seen* attributes and objects during training.
- 🔄 **Structure-guided Adaptation Strategy (SAS)** — adaptively aligns *unseen* concepts with their semantically similar *seen* ones at inference.

> 🧠 In short: SPA keeps CLIP’s semantic geometry intact while improving generalization from seen to unseen compositions — with minimal extra cost.

## 📁 Dataset Preparation
SPA follows the standard OV-CZSL datasets:
- MIT-States
- C-GQA
- VAW-CZSL

## ⚙️ Usage
The SPA framework depends on the following main requirements:
- torch==1.12.1+cu113
- Transformers 4.45.1
- OpenCV 4.10.0
- tqdm

### How to Run (take MIT-States for example)
```python
python train.py --cfg config/mit.yml
```
