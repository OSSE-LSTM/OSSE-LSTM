
# Dual-Stream OSSE-LSTM for Interpretable Few-Shot Time Series Classification

This repository implements **Dual-Stream OSSE-LSTM**, an episodic metric-learning framework for few-shot time series classification. The model combines an **Omni-Scale CNN with Squeeze-and-Excitation blocks** for multi-scale local motif extraction and a **Bidirectional LSTM** for temporal context modeling. Classification is performed with **Prototypical Networks**, where each class is represented by the mean embedding of its support samples.

The repository also includes a **Counterfactual Integrated Gradients (C-IG)** module for interpreting prototype-based decisions and a **C-IG-guided test-time augmentation (TTA)** procedure for refining support prototypes during inference.

---

## Architecture Overview

<img width="10910" height="6610" alt="architecture" src="https://github.com/user-attachments/assets/00872fb6-4527-4bf6-bc96-d273fd0f4620" />

The full pipeline consists of two main phases.

### Phase I: Episodic Training and Prototype-Based Classification

Given an input time series, the model creates few-shot episodes containing a support set and a query set. The support samples are used to compute class prototypes, while query samples are classified by their Euclidean distance to these prototypes in the learned embedding space.

The encoder contains two parallel streams:

- **OS-CNN + SE stream**  
  Extracts multi-scale local temporal motifs using omni-scale convolutional filters. Squeeze-and-Excitation blocks recalibrate feature channels before temporal pooling.

- **BiLSTM stream**  
  Models global temporal context by processing the sequence in both forward and backward directions.

The two streams are independently `L2`-normalized, concatenated, passed through Layer Normalization, and finally projected again onto a normalized embedding space.

### Phase II: Counterfactual Interpretation and Test-Time Refinement

After training, C-IG explains the learned prototype space by comparing target class representatives against opposing class representatives. It identifies temporal regions that increase separation between prototypes. These regions are interpreted as **neural shapelet regions**.

The same attribution maps can also guide test-time prototype refinement. Instead of perturbing the entire support signal, the method preserves high-attribution regions and injects small noise mainly into low-attribution background regions.

---

## Repository Workflow

The standard workflow is:

1. Split datasets into training ratios.
2. Train OSSE-LSTM with episodic Prototypical Networks.
3. Run C-IG interpretation.
4. Run C-IG-guided test-time prototype refinement.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook
- Other dependencies listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
````

---

## Dataset Preparation

Before training, run the dataset splitting notebook:

```bash
jupyter notebook split_dataset.ipynb
```

This notebook loads the original UCR/UEA time series datasets and saves ratio-based splits into:

```text
Splitted_datasets/
└── <DatasetName>/
    └── <ratio>/
        └── <split_index>/
            └── <DatasetName>.npy
```

Each saved `.npy` file contains:

```python
{
    "X_train": ...,
    "y_train": ...,
    "X_test": ...,
    "y_test": ...
}
```

For univariate datasets, the input shape is converted to:

```text
[N, 1, L]
```

For multivariate datasets, the input shape is:

```text
[N, C, L]
```

where `N` is the number of samples, `C` is the number of channels, and `L` is the sequence length.

---

## Training

To train the model in a 2-shot setting, run:

```bash
bash train_proto.sh 0 2 os_cnn ./train.py 2
```

Example meaning of the command:

```text
0          GPU index
2          number of support samples / shot setting
os_cnn     architecture name used by the logger and checkpoint path
./train.py training script
2          query/support configuration passed to the script
```

Please check `train_proto.sh` if you modify the argument order.

During training, the model follows episodic learning:

1. Sample an `N`-way `K`-shot episode.
2. Encode all support and query samples using OSSE-LSTM.
3. Compute class prototypes from support embeddings.
4. Classify query samples by negative Euclidean distance to prototypes.
5. Update the encoder by cross-entropy loss over query predictions.

Model checkpoints are saved under the configured log directory, usually in a structure similar to:

```text
logs/
└── os_cnn/
    └── <DatasetName>/
        └── <ratio>/
            └── <split_index>/
                └── <run_id>/
                    └── baseline_classifier/
                        └── model_os_cnn_best.pth
```

---

## Interpretability Analysis

After training, run:

```bash
python ./interpretability.py
```

This script loads a trained checkpoint and applies **Counterfactual Integrated Gradients (C-IG)** to extract class-specific temporal attribution maps.

C-IG works by:

1. Computing a class prototype in the learned embedding space.
2. Selecting the nearest real sample to the prototype as the class representative.
3. Building counterfactual paths between target and opposing class representatives.
4. Integrating prototype-margin gradients along these paths.
5. Producing temporal attribution maps.
6. Extracting stable neural shapelet regions from high-attribution intervals.

The resulting visualizations are saved under:

```text
saved_visualizations/
└── <DatasetName>/
```

Important path settings to check inside `interpretability.py`:

```python
DATASET = "MoteStrain"
DATASET_PATH = "C:/Users/ADMIN/Desktop/OSSE-LSTM/OSCNN_splitted_datasets"
LOG_ROOT = PROJECT_ROOT / "logs" / "os_cnn" / DATASET / "1" / "10"
```

Update these paths according to your local dataset and checkpoint directory.

---

## Test-Time Prototype Refinement

To run C-IG-guided test-time augmentation, use:

```bash
bash train_proto.sh 0 2 os_cnn ./test_time_aug.py 2
```

This evaluates the trained model with prototype refinement at inference time.

The procedure is:

1. Compute standard prototypes from the support set.
2. Generate C-IG masks for support samples.
3. Preserve high-attribution regions.
4. Add small noise mainly to low-attribution background regions.
5. Recompute class prototypes using both original and augmented support embeddings.
6. Classify queries using refined prototypes.

No model parameters are updated during this process.

---

## Important Hyperparameters

### OS-CNN Receptive Field

The maximum receptive field is automatically bounded by:

```python
receptive_field_shape = min(int(sequence_length / 4), 89)
```

This prevents the convolutional kernels from becoming too large relative to the input sequence length.

### OS-CNN Parameter Budget

The OS-CNN structure is generated using:

```python
layer_parameter_list = generate_layer_parameter_list(
    1,
    receptive_field_shape,
    [8 * 128 * num_channels, 5 * 128 * 256 + 2 * 256 * 128],
    num_channels
)
```

For univariate datasets, `num_channels = 1`.

For multivariate datasets, `num_channels > 1`, so the first parameter term should scale with the number of channels. This ensures that the first convolutional stage has enough capacity to process multi-channel inputs.

Example for a 12-channel dataset:

```python
layer_parameter_list = generate_layer_parameter_list(
    1,
    receptive_field_shape,
    [8 * 128 * 12, 5 * 128 * 256 + 2 * 256 * 128],
    12
)
```

Practical note:

* Increase the first parameter budget when using multivariate datasets.
* Keep the output channel size reasonable to avoid unstable training.
* A target output channel size around a few hundred is usually sufficient.

### BiLSTM Settings

The BiLSTM branch uses:

```python
lstm_hidden_dim = 16
bidirectional = True
dropout = 0.5
```

Since the LSTM is bidirectional, the recurrent vector has dimension:

```text
2 * lstm_hidden_dim = 32
```

Dropout is applied only to the LSTM representation before fusion.

### Feature Fusion

The CNN and LSTM representations are normalized separately:

```python
cnn_feat = F.normalize(cnn_feat, p=2, dim=1)
lstm_feat = F.normalize(lstm_feat, p=2, dim=1)
```

They are then concatenated, passed through Layer Normalization, and normalized again:

```python
combined_feat = torch.cat([cnn_feat, lstm_feat], dim=1)
combined_feat = final_norm(combined_feat)
combined_feat = F.normalize(combined_feat, p=2, dim=1)
```

This keeps the two streams on comparable scales before Euclidean prototype comparison.

---

## C-IG TTA Parameters

In `test_time_aug.py`, the main parameters are:

```python
N_AUG = 5
CIG_STEPS = 20
NOISE_SCALE = 0.1
```

Meaning:

* `N_AUG`: number of augmented support variants per support sample.
* `CIG_STEPS`: number of interpolation steps for C-IG mask generation.
* `NOISE_SCALE`: strength of the Gaussian perturbation applied to low-attribution regions.

For perturbation-sensitive datasets such as `Wine`, use a smaller value:

```python
NOISE_SCALE = 0.01
```

The noise is scaled by the temporal standard deviation of each support sample:

```python
noise = torch.randn_like(original_support) * signal_std * NOISE_SCALE
augmented_support = original_support + noise * (1.0 - cam)
```

Thus, high-attribution regions are preserved, while low-attribution background regions are softly perturbed.

---

## Notes on Univariate and Multivariate Data

### Univariate Datasets

No special modification is required. The dataset loader automatically converts inputs from:

```text
[N, L]
```

to:

```text
[N, 1, L]
```

### Multivariate Datasets

For multivariate `.ts` files, each sample is loaded as:

```text
[C, L]
```

If the sequences have different lengths, the loader pads them to the maximum length within the dataset.

When training on multivariate datasets, check:

```python
train.data.shape[1]
```

This gives the number of channels and should be passed into both:

```python
generate_layer_parameter_list(..., in_channel=train.data.shape[1])
OS_CNN(..., in_channels=train.data.shape[1])
```

---

## Output Files

Typical outputs include:

```text
logs/
    Training logs and checkpoints

test_accuracy_log_all_as_proto_<K>shot/
    Final test accuracy logs

saved_visualizations/
    C-IG attribution maps and extracted shapelet visualizations

Splitted_datasets/
    Ratio-based train/test splits
```

---

## Common Issues

### 1. Checkpoint path not found

Make sure that the dataset name, ratio, split index, and architecture name match the trained checkpoint folder.

Example:

```python
LOG_ROOT = Path(args.log_dir) / name / str(target_ratio) / str(target_ind)
model_best_path = subdir / "baseline_classifier" / f"model_{args.arch}_best.pth"
```

### 2. Dataset path not found

Update:

```python
args.dataset_root
DATASET_PATH
```

to match the location of `OSCNN_splitted_datasets`.

### 3. CUDA unavailable

Training requires a CUDA-enabled GPU. Check:

```python
torch.cuda.is_available()
```

### 4. Multivariate shape mismatch

Verify that all samples are formatted as:

```text
[N, C, L]
```

and that `in_channels` is set to the correct channel number.

---

## Recommended Running Order

```bash
# 1. Split datasets into ratio-based files
jupyter notebook split_dataset.ipynb

# 2. Train OSSE-LSTM under 2-shot setting
bash train_proto.sh 0 2 os_cnn ./train.py 2

# 3. Run C-IG interpretability analysis
python ./interpretability.py

# 4. Run C-IG-guided test-time prototype refinement
bash train_proto.sh 0 2 os_cnn ./test_time_aug.py 2
```

---

