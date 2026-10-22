# Contributing & Development Workflow (Step 10)

This repository follows standard Git branching and GitHub Pull Request conventions as outlined in the **Batch F Capstone Guidelines**.

---

## 1. Branch Naming Conventions
- `main`: Production-ready, stable codebase.
- `feature/dataset-pipeline`: Data collection, augmentation, and loaders.
- `feature/cnn-baseline`: Initial custom convolutional model development.
- `feature/resnet-transfer-learning`: Residual transfer learning implementation.
- `feature/vision-transformer`: ViT architecture, self-attention heads, and fine-tuning.
- `feature/presentation-and-docs`: Slide deck, reports, and documentation.

---

## 2. Pull Request (PR) Checklist
When submitting a Pull Request:
1. Ensure the code executes without unhandled exceptions.
2. Confirm that data loaders maintain non-overlapping train/val/test splits (zero data leakage).
3. Update relevant evaluation metrics in `results/` if model weights or hyperparameters were altered.
4. Provide a descriptive PR summary linking the corresponding GitHub Issue (e.g., `Resolves #3`).

---

## 3. GitHub Issues Framework (Step 9)
The project lifecycle is tracked via 5 primary issues:
- **Issue #1:** Dataset Collection and Stratified Preprocessing
- **Issue #2:** Custom CNN Baseline Model Development
- **Issue #3:** ResNet18 Transfer Learning and Regularization
- **Issue #4:** Vision Transformer (ViT-B/16) Architecture & Fine-Tuning
- **Issue #5:** Capstone Deliverables, Presentation, and SOTA Verification
