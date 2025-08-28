## 💾 Pretrained Models

Below is the list of available pretrained models. Each checkpoint is trained under distinct noise configurations.

| Model Name | Description |
| :--------: | :---------- |
| idf_g_15.ckpt (default) | Trained with additive Gaussian noise (σ = 15). This default configuration matches the experimental setup described in the paper. |
| idf_g_15_50.ckpt | Trained with additive Gaussian noise where σ is uniformly sampled from the range [15, 50]. |
| idf_gp.ckpt | Trained with a combination of additive Gaussian noise (σ in [15, 50]) and Poisson noise (λ in [1.0, 4.0]). |

