# Variational Autoencoder (VAE) Results

### Model Loss Summary - follows the Hands On Machine Learning implementation (Geron, 2022) with additional logic for masking
1. Random Masking Layer

    This layer randomly selects one of the four features (DNA methylation, H3K9me3, H3K27me3, or gene expression) and applies a mask to it (-1).
    The resulting masked input (a 12001 binary vector with a masked section using -1) is then passed through the VAE.

2. KL Divergence Loss

    The KLDivergenceLayer computes how much the learned latent distribution (represented by codings_mean and codings_log_var) diverges from the standard normal distribution (which serves as a prior).
    The loss is normalised by the number of features (12001) to ensure that the scale is appropriate.

3. Custom Loss Layer

    This layer calculates the reconstruction loss specifically for the masked sections.
    Masking Logic: The model first identifies indices that were masked (i.e. where the input was set to -1.0).
    Binary Cross-Entropy Loss: The reconstruction loss is computed using binary cross-entropy since the data is binary.
        The binary cross-entropy loss is calculated between the original inputs and the reconstructed outputs.
        The reconstructed outputs are clipped to avoid issues with taking the log of zero or negative numbers, which could lead to numerical instability.
    Masked Loss: The loss is then focused only on the masked indices by multiplying the loss with the mask indices.
    Averaging and Avoiding Division by Zero:
        The loss is averaged over the masked indices to compute a per-sample loss.


### Baseline model: Using all features (and all data) as input without masking
(481647)
Start with a baseline - use all data to input and regenerate 12001 input (4000 DNAm, 4000 K9, 4000 K27, 1 Expression)

`codings_size = 50`

![VAE V1 Loss plot](coding_size50/vae_v1.png)

Notes:
- Not entirely confident how to interpret this - assuming this is good since 0 is perfect and 1 would be random(?)

### Split expression model: Splitting the data into two groups (silent and non-silent) as input without masking

#### Version 1: Training 2 models separately
(481698)

![VAE Split V1 Loss plot](coding_size50/vae_silent_vs_non_silent.png)

Notes:
- Silent class is easier to regenerate than non-silent (more variance in this class)
- Separating the two gives the lowest reconstruction yet for the silent class
- This makes sense that it would be harder to regenerate the non-silent class as it represents all expression values except for 0 (thus more variation)

#### Version 2: Training 1 model (same as baseline model) and validating classes separately
(485904)

![VAE Baseline Split V1 Loss plot](coding_size50/vae_baseline_silent_vs_non_silent.png)

Notes:
- Supports previous findings - silent is easier to recreate (which aligns with the logic that silent has less variation, due to 'zero vs non-zero' nature)



### Random Masking model: using all features (and all data) as input with randomly masked subsections
(482584)
This model utilises a random masking function which randomly selects a feature subset for each minibatch and validation set and masks this whole feature as -1. It then finds the loss of the masked section - so the below plot represents the error of regenerating the masked section. 

![VAE Random Mask All Data](coding_size300/vae_loss_plot_allcat_300_v4.png)
![VAE Random Mask All Data (Log)](coding_size300/vae_loss_plot_allcat_300_v4_log.png)


Notes:
- Inconsistent loss curve due to random masking - appeears the model is struggling to learn with such variance
- This makes sense since the reconstruction loss is based solely on the masked region whereas the KL Loss comes from the entire input


### Fixed Masking models: using all features but training the model whilst masking only one section throughout training and validation

Initially, this was separated out into individual models to better understand the spikes and components that the Random Masking model is required to learn.

To further investigate the reconstruction of individual components (in particular, K9 and K27, which we currently are unsure whether the model is predicting well or just predicting zero the whole time), we separate loss into KL Loss and reconstrcution loss. This will show whether the loss is derived from the guassians or reconstruction. 

In VAE, total loss is composed of two components:
1. Reconstruction loss: How well the model can reconstruct the data from the latent space. It ensures the decoder output is close to the original input. 
2. KL Divergence Loss: How much the learned latent distribution deviates from a normal Guassian distribution - this regularises the latent space by encouraging encoded variables to follow Guassian distribution. 

What does this show:
1. Reconstruction loss: if low, the model can recreate the input well but if too low, may indicate overfitting and poor generalisation
2. KL Loss: if low, then the model is learning a smooth latent space close to Guassian, but very low loss might indicate underfitting.

Plotting both allows us to track the trade-off between regularisation and accuracy.

Implications for sparse K9 and K27:
- To ensure the model isn't taking advantage of the fact that the dataset is comprised of mostly zeros, we can check KL loss
- If KL loss is very low, this would indicate the model isn't learning the data distribution and just using Guassian distribution   
- Reconstruction loss will likely be low regardless of the models predictive ability due to the data's sparsity
- If both are low, then liekly the model is underfitting (predicting zeros too often)

KL loss is universal to the whole input, whilst reconstruction loss can be specific to the masked input section. As such, we will track:
- masked total loss
- whole input total loss
- masked reconstruction loss
- whole input reconstruction loss
- KL loss

#### Masking DNAm
![VAE Fixed Mask DNAm](coding_size100/vae_loss_plot_DNAm_100_v4.png)
![VAE Fixed Mask DNAm (Log)](coding_size100/vae_loss_plot_DNAm_100_v4_log.png)

Notes:
- Training loss rapidly drops after first epoch - relatively stable after this
- Reconstruction loss is fairly low (validation less than training? Is this just because the model is updating weights then testing?)
- KL loss is extremely low, with a small discrepancy between training and validation (contributes practically nothing to total loss)
- Does this mean the model isn't learning a meaningful latent representation of the inputs since the KL loss is so low?

| Metric | Training | Validation |
| ------------- | ------------- | ------------- |
| Total Loss | 0.1459 | 0.1457 |
| Reconstruction Loss | 0.1459 | 0.1457 |
| KL Loss | 5.9530e-08 | 1.0244e-05 |
(488146)

#### Masking K9
![VAE Fixed Mask K9](coding_size100/vae_loss_plot_K9_100_v4.png)
![VAE Fixed Mask K9 (log)](coding_size100/vae_loss_plot_K9_100_v4_log.png)


Notes:
- Not much continual learning here - plateaus very early
- Again, KL loss is basically 0 which is concerning?


| Metric | Training | Validation |
| ------------- | ------------- | ------------- |
| Total Loss | 0.1422 | 0.1463 |
| Reconstruction Loss | 0.1422 | 0.1463 |
| KL Loss | 1.4997e-06 | 1.3168e-06 |
(488147)



#### Masking K27
![VAE Fixed Mask K27](coding_size100/vae_loss_plot_K27_100_v4.png)
![VAE Fixed Mask K27 (log)](coding_size100/vae_loss_plot_K27_100_v4_log.png)

Notes:
- Not much continual learning here - plateaus very early (even lower than K9 - probably because even more sparse)
- Again, KL loss is basically 0 (no meaningful learning?)


| Metric | Training | Validation |
| ------------- | ------------- | ------------- |
| Total Loss | 0.0607 | 0.0613 |
| Reconstruction Loss | 0.0607 | 0.0613 |
| KL Loss | 5.2776e-07 | 7.5919e-07 |
(488149)



#### Masking Expression
![VAE Fixed Mask Expression](coding_size100/vae_loss_plot_expression_100_v4.png)
![VAE Fixed Mask Expression (Log)](coding_size100/vae_loss_plot_expression_100_v4_log.png)


Notes:
- Not much continual learning here - plateaus very early (even lower than K9 - probably because even more sparse)
- Again, KL loss is basically 0 (no meaningful learning?)


| Metric | Training | Validation |
| ------------- | ------------- | ------------- |
| Total Loss | 0.6651 | 0.6610 |
| Reconstruction Loss | 0.6651 | 0.6610 |
| KL Loss |  9.4815e-07 | 1.0311e-06 |
(488150)


### All-pairs approach

#### DNAm
![VAE All Pairs DNAm](coding_size100/all_pairs/vae_loss_plot_DNAm_all_pairs_v4.png)
![VAE All Pairs DNAm (Log)](coding_size100/all_pairs/vae_loss_plot_DNAm_all_pairs_v4_log.png)

#### K9
![VAE All Pairs K9](coding_size100/all_pairs/vae_loss_plot_K9_all_pairs_v4.png)
![VAE All Pairs K9 (Log)](coding_size100/all_pairs/vae_loss_plot_K9_all_pairs_v4_log.png)

#### K27
![VAE All Pairs K27](coding_size100/all_pairs/vae_loss_plot_K27_all_pairs_v4.png)
![VAE All Pairs K27 (Log)](coding_size100/all_pairs/vae_loss_plot_K27_all_pairs_v4_log.png)

#### Expression
![VAE All Pairs Expression](coding_size100/all_pairs/vae_loss_plot_expression_all_pairs_v4.png)
![VAE All Pairs Expression (Log)](coding_size100/all_pairs/vae_loss_plot_expression_all_pairs_v4_log.png)


Notes:

- Overall, seeing similar issues here with extremely low KL Loss (basically zero) for all plots
- Will have to reattempt this with much smaller coding size

## Investigating Low KL Loss

**Attempt to reduce coding size down to 10** (503589)

![VAE Fixed Mask DNAm](coding_size10/vae_loss_plot_DNAm_10_v4.png)
![VAE Fixed Mask DNAm (Log)](coding_size10/vae_loss_plot_DNAm_10_v4_log.png)

**Attempt to reduce coding size down to 10 with weighted KL (1.2)** (503590)
![VAE Fixed Mask DNAm](coding_size10/vae_loss_plot_DNAm_10_v5.png)
![VAE Fixed Mask DNAm (Log)](coding_size10/vae_loss_plot_DNAm_10_v5_log.png)

**Recreate no masking model but tracking KL Loss (coding_size=50)** (503591)
![VAE with KL Loss](coding_size50/KLLoss/vae_v1_total_and_kl_loss_50.png)

- the loss and KL Divergence here looks better?


Could it be an issue with masking then?
Try:
- random masking within section
- training model on whole recon loss + KL
- conditional VAE