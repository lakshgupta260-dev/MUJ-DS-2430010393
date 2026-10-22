# Capstone Project: Automated Tomato Disease Classification

**Batch:** Batch F  
**Track:** Computer Vision & Deep Learning in Smart Agriculture  
**Author:** Laksh Gupta (Reg: 2430010393)  

---

## 1. Executive Summary
This Capstone Project develops an automated diagnostic system for tomato leaf pathologies using deep learning. Through an empirical 3-stage architectural study, we evaluate:
1. **Custom 3-Block CNN (From Scratch):** 72.20% Test Accuracy
2. **Fine-Tuned ResNet18 (Residual CNN):** 97.20% Test Accuracy
3. **Vision Transformer ViT-B/16 (Proposed SOTA):** 98.40% Test Accuracy

The proposed Vision Transformer model utilizes Multi-Head Self-Attention across 196 non-overlapping image patches to capture long-range contextual dependencies and fine-grained necrosis patterns across tomato foliage.

## 2. Team & Collaborative Integration (Step 7 & 8)
- **Model Development & Fine-Tuning:** Laksh Gupta (Lead Developer)
- **Data Pipeline & Augmentation:** Implemented stratified sampling and tensor transformations.
- **Evaluation & Benchmarking:** Conducted SOTA comparison across 13 published studies (2020–2026).
- **Deployment Design:** Developed REST API specification and ONNX inference export.

## 3. GitHub Issues & Pull Requests Roadmap (Step 9 & 10)
- **Issue #1:** Dataset Collection, Curation, and Stratified Partitioning (Closed via PR #1)
- **Issue #2:** Custom CNN Baseline Model Development & Regularization (Closed via PR #2)
- **Issue #3:** ResNet18 Transfer Learning & Hyperparameter Tuning (Closed via PR #3)
- **Issue #4:** Vision Transformer (ViT-B/16) Architecture & Attention Optimization (Closed via PR #4)
- **Issue #5:** Deliverables, Documentation, Presentation & SOTA Verification (Closed via PR #5)
