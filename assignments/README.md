# Batch F Capstone Project Assignments & Milestones

**Student Name:** Laksh Gupta  
**Registration Number:** 2430010393  
**Branch:** Data Science (MUJ-DS)  
**Batch:** Batch F  

---

## Weekly Progression & Milestone Log

### Week 1: Problem Definition & Literature Review
- Selected Agricultural AI domain: Tomato Leaf Disease Classification.
- Reviewed baseline research papers (Trivedi et al. 2021, Shrivastava et al. 2020).
- Downloaded and analyzed the PlantVillage benchmark dataset.

### Week 2: Data Preprocessing & Exploratory Data Analysis
- Analyzed 10 classes (9 disease categories + 1 healthy control).
- Verified uniform class distribution (1,000 images per class, total 10,000 images).
- Formulated stratified 70/15/15 train/val/test split.
- Designed data augmentation pipeline (rotation, flipping, color jitter).

### Week 3: Baseline Custom CNN Architecture
- Designed a lightweight 3-block convolutional neural network from scratch.
- Total parameters: 120,458 (~120K).
- Achieved baseline accuracy of 72.20%.
- Documented key limitations: shallow feature hierarchy and local receptive fields.

### Week 4: Research Gap Analysis & Transfer Learning (ResNet18)
- Performed comprehensive SOTA benchmarking against IEEE/Elsevier publications.
- Upgraded model architecture to Fine-Tuned ResNet18 (11.1M parameters).
- Implemented combined L1/L2 regularization and ReduceLROnPlateau scheduling.
- Validation accuracy jumped to 97.27%, test accuracy to 97.20% (+25.0% boost).

### Week 5: Vision Transformer (ViT-B/16) Implementation
- Formulated proposed SOTA architecture: Vision Transformer (ViT-B/16).
- 16x16 patch partitioning, 196 tokens, 12 transformer encoder blocks, 12 attention heads.
- Applied AdamW optimizer with CosineAnnealingLR and label smoothing.
- Achieved state-of-the-art Test Accuracy: **98.40%** (Validation: **98.60%**).
- Macro F1-score: **0.98** across all 10 pathology classes.

### Week 6: Model Evaluation, Packaging & Deliverables
- Generated full Confusion Matrix, ROC-AUC curves, and classification reports.
- Prepared 12-slide Capstone presentation (`Tomato_Disease_ViT_Capstone_Presentation.pptx`).
- Finalized GitHub repository organization conforming to Batch F guidelines.
