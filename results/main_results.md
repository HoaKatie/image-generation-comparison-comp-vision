# Main results: fashion_mnist, latent_dim=32, VAE beta=1

Mean ± std over seeds. The last row runs the same metrics on 10,000 real TRAINING images instead of generated ones (`python -m src.real_baseline`): the best score a perfect generator could expect. Evaluation classifier test accuracy = 0.907.

| model | seeds | FID-Inception ↓ | FID-classifier ↓ | Precision-Inc ↑ | Recall-Inc ↑ | Precision-clf ↑ | Recall-clf ↑ | Modes covered ↑ | Class KL ↓ | Laplacian var | HF ratio | Interp. CV ↓ | NN ratio | Recon PSNR ↑ | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DCGAN | 3 | 13.3 ± 0.52 | 1.29 ± 0.047 | 0.581 ± 0.022 | 0.495 ± 0.0059 | 0.852 ± 0.0077 | 0.854 ± 0.0012 | 10 ± 0 | 0.0215 ± 0.0052 | 0.146 ± 0.0014 | 0.1 ± 0.0039 | 0.691 ± 0.021 | 1.07 ± 0.028 | – | 394 ± 56 |
| VAE | 3 | 56.9 ± 1.6 | 20.7 ± 1.2 | 0.222 ± 0.0038 | 0.141 ± 0.0055 | 0.695 ± 0.0075 | 0.489 ± 0.0067 | 10 ± 0 | 0.217 ± 0.01 | 0.038 ± 0.0011 | 0.0335 ± 0.001 | 0.509 ± 0.0074 | 1 ± 0.026 | 22.4 ± 0.079 | 223 ± 25 |
| Real data (train vs. test) | – | 2.4 | 0.0339 | 0.782 | 0.779 | 0.903 | 0.893 | 10 | 0.0051 | 0.15 | 0.104 | – | – | – | – |
