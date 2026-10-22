# System Architecture & Deployment Blueprint

```
+-----------------------------------------------------------------------------------+
|                            SMART AGRICULTURE SYSTEM                               |
+-----------------------------------------------------------------------------------+
                                        |
                   [ Farmer Smartphone / Drone Camera ]
                                        |  (Uploads Leaf Image)
                                        v
+-----------------------------------------------------------------------------------+
|                           FASTAPI INFERENCE BACKEND                               |
+-----------------------------------------------------------------------------------+
|  1. Image Ingestion (224 x 224 x 3 RGB)                                            |
|  2. TorchVision Transform: Tensor Normalization (ImageNet mean & std)             |
|  3. Model Execution Engine:                                                       |
|     - Option A: Proposed Vision Transformer (ViT-B/16) -> 98.40% Accuracy         |
|     - Option B: Mobile Edge ResNet18 -> 97.20% Accuracy (Low Latency / Offline)   |
|  4. Softmax Probability Distribution across 10 Classes                             |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                               OUTPUT DIAGNOSTICS                                  |
+-----------------------------------------------------------------------------------+
|  - Predicted Disease Label (e.g., "Late Blight", "Bacterial Spot")                 |
|  - Confidence Score (e.g., 99.4%)                                                 |
|  - Recommended Treatment / Agronomic Pesticide Protocol                           |
|  - Attention Rollout Heatmap (Displays diagnosed leaf regions)                    |
+-----------------------------------------------------------------------------------+
```

## Architectural Specifications
- **Input Dimensions:** $224 \times 224 \times 3$
- **Patch Embedding:** $16 \times 16 \times 3 \to 768$ (196 visual tokens)
- **Attention Heads:** 12 Multi-Head Attention blocks
- **Depth:** 12 Transformer Encoder layers
- **Total Parameters:** $86,575,882$
- **Quantization Compatibility:** Model can be exported to ONNX format with INT8 quantization, compressing weight footprint from 330 MB to 85 MB for on-device mobile execution.
