# Key Research Questions: NLOM Thyroid Image Preprocessing

## Normalization
1. How should percent normalization be adapted for NLOM images given their Poisson noise distribution?
2. What intensity scaling approaches best preserve subcellular features in harmonic generation microscopy?
3. How do normalization parameters differ between SHG and THG microscopy modalities?

## CLAHE Optimization
4. What CLAHE parameters (tile size, clip limit) are optimal for enhancing subcellular structures?
5. How should CLAHE be adapted for z-stack NLOM images versus single-plane acquisitions?
6. What are the tradeoffs between CLAHE and other contrast enhancement methods for NLOM?

## Additional Preprocessing
7. What denoising techniques are most effective for Poisson noise in NLOM?
8. How can we mitigate photobleaching artifacts in fluorescence-based NLOM?
9. What edge enhancement methods improve segmentation of thyroid follicles?

## Validation
10. What quantitative metrics best measure preprocessing effectiveness for NLOM?
11. How should we validate preprocessing impact on downstream classification tasks?
12. What visual assessment protocols exist for NLOM preprocessing quality?