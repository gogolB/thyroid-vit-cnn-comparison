# Primary Findings: NLOM Thyroid Preprocessing

## Normalization Recommendations
1. **Percentile Normalization**
   - Use 1st-99th percentile scaling to mitigate Poisson noise extremes
   - Formula: 
     ```
     I_normalized = (I - P1) / (P99 - P1)
     ```
   - PyTorch Implementation:
     ```python
     def percentile_normalization(tensor: torch.Tensor, low=1, high=99):
         p_low = torch.quantile(tensor, low/100)
         p_high = torch.quantile(tensor, high/100)
         return (tensor - p_low).clamp(0) / (p_high - p_low + 1e-7)
     ```

2. **Poisson Noise Adaptation**
   - Apply Anscombe transform before normalization:
     ```python
     def anscombe(x):
         return 2 * torch.sqrt(x + 3/8)
     ```

## CLAHE Optimization
| Parameter      | Recommendation | Rationale |
|----------------|----------------|-----------|
| Tile Size      | 32x32          | Matches subcellular feature scale |
| Clip Limit     | 0.03           | Balances enhancement and noise |
| Channel Handling | Per-channel  | Preserves spectral characteristics |

```python
# PyTorch CLAHE implementation
clahe = torchvision.transforms.RandomApply([
    torchvision.transforms.RandomAffine(0),
    torchvision.transforms.RandomApply([
        torchvision.transforms.Lambda(
            lambda x: equalize_adapthist(x.numpy(), clip_limit=0.03)
        )
    ], p=0.8)
], p=0.5)
```

## Additional Processing Steps
1. **Z-stack Alignment**
   - Phase correlation-based registration
2. **Photobleaching Correction**
   - Exponential decay modeling per z-slice
3. **Edge Enhancement**
   - Multiscale Laplacian filtering

## Validation Metrics
1. **Quality Metrics**
   - SSIM (Structural Similarity)
   - CNR (Contrast-to-Noise Ratio)
2. **Task Performance**
   - Segmentation Dice score
   - Classification accuracy