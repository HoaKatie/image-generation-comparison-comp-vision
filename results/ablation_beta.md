# Ablation: VAE beta (fashion_mnist)

Mean ± std over seeds.

| beta | seeds | FID-Inception ↓ | FID-classifier ↓ | Precision-Inc ↑ | Recall-Inc ↑ | Precision-clf ↑ | Recall-clf ↑ | Modes covered ↑ | Class KL ↓ | Laplacian var | HF ratio | Interp. CV ↓ | NN ratio | Recon PSNR ↑ | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1 | 60.1 | 22 | 0.171 | 0.229 | 0.641 | 0.613 | 10 | 0.216 | 0.0408 | 0.0386 | 0.445 | 1.1 | 23.4 | 197 |
| 1.0 | 3 | 56.9 ± 1.6 | 20.7 ± 1.2 | 0.222 ± 0.0038 | 0.141 ± 0.0055 | 0.695 ± 0.0075 | 0.489 ± 0.0067 | 10 ± 0 | 0.217 ± 0.01 | 0.038 ± 0.0011 | 0.0335 ± 0.001 | 0.509 ± 0.0074 | 1 ± 0.026 | 22.4 ± 0.079 | 223 ± 25 |
| 4.0 | 1 | 74.1 | 24 | 0.245 | 0.0126 | 0.715 | 0.142 | 10 | 0.237 | 0.0284 | 0.022 | 0.627 | 0.795 | 20.1 | 197 |
