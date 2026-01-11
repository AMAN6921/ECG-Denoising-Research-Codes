# ECG Denoising Project

Advanced ECG signal denoising using hybrid VMD (Variational Mode Decomposition) algorithms.

## Outcome

This project achieves high-quality ECG signal denoising with the following performance levels:

- **Input SNR**: 11.8 dB (noisy ECG signal)
- **Output SNR**: Up to 50 dB (clean denoised signal)
- **SNR Improvement**: ~38 dB enhancement

## Output

The algorithms produce:
- Denoised ECG signals with preserved morphology
- Performance metrics (SNR, PSNR, correlation)
- Comparison results across multiple MIT-BIH records

## Files

### Completed Versions (Production Ready)
- `hybrid_vmd_denoiser_40db_final.py` - 40 dB target performance
- `hybrid_vmd_denoiser_49db_final.py` - 49 dB target performance  
- `hybrid_vmd_denoiser_50db_mit_final.py` - 50 dB target performance (best)

### Development Versions
- `ECG-Denoising-Attempt/` - Various experimental implementations

## Usage

```python
python hybrid_vmd_denoiser_50db_mit_final.py
```

## Requirements

- numpy, scipy, pywt
- vmdpy (for VMD algorithm)
- wfdb (for MIT-BIH database access)

## Performance

Best achieved results on MIT-BIH data:
- 99.8% achievement rate of 50 dB target
- Correlation coefficient > 0.99
- Preserves ECG morphological features