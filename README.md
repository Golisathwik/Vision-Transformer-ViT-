# Vision Transformer (ViT)

## Overview

This project demonstrates a Vision Transformer (ViT) implementation using TensorFlow / Keras on the CIFAR-10 dataset. The notebook code explicitly loads CIFAR-10 via `keras.datasets.cifar10.load_data()` and then builds, trains, and evaluates a simplified ViT model.

The implementation includes:
- CIFAR-10 data loading and preprocessing
- Image resizing and patch extraction
- Patch encoding with learned position embeddings
- Transformer encoder blocks with multi-head self-attention
- Classifier head with dense layers for final prediction
- Training using AdamW optimizer and checkpoint saving
- Evaluation using accuracy and top-5 accuracy
- Image prediction helper for sample inference

## Key Notebook Workflow

1. Install required packages:
   - `tensorflow==2.13.0`
   - `numpy==1.24.3`
   - `protobuf==3.20.3`
   - `tensorflow-addons==0.21.0`

2. Import TensorFlow and supporting libraries.

3. Load CIFAR-10 dataset:
   - `keras.datasets.cifar10.load_data()`
   - `num_classes = 10`
   - `input_shape = (32, 32, 3)`

4. Restrict the dataset to a small subset for faster experimentation:
   - `x_train = x_train[:500]`
   - `y_train = y_train[:500]`
   - `x_test = x_test[:500]`
   - `y_test = y_test[:500]`

5. Set ViT parameters:
   - `learning_rate = 0.001`
   - `weight_decay = 0.0001`
   - `batch_size = 256`
   - `num_epochs = 40`
   - `image_size = 72`
   - `patch_size = 6`
   - `num_patches = (image_size // patch_size) ** 2`
   - `projection_dim = 64`
   - `num_heads = 4`
   - `transformer_layers = 8`
   - `mlp_head_units = [2048, 1024]`

6. Define augmentation and ViT building blocks:
   - Data augmentation using `keras.Sequential`
   - Custom `Patches` layer to split images into patches
   - `PatchEncoder` layer for patch projection and positional embeddings
   - `mlp()` helper for dense blocks

7. Build the ViT classifier in `create_vit_classifier()`:
   - Input layer
   - Patch extraction and encoding
   - Transformer encoder stack
   - Flatten + dropout
   - Final dense classifier output

8. Train and evaluate the model using `run_experiment(model)`:
   - `tfa.optimizers.AdamW`
   - `SparseCategoricalCrossentropy(from_logits=True)`
   - Metrics: `accuracy`, `top_5_accuracy`
   - Checkpoint callback saved to `./tmp/checkpoint`
   - Model evaluation on test subset

9. Run sample prediction using `img_predict(images, model)`.

## Results

The notebook records model evaluation and sample predictions. The static HTML export shows training output and evaluation logs, including test accuracy and top-5 accuracy values. One sample run produced a test accuracy around `34.4%` and a test top-5 accuracy around `84.8%` on the selected subset.

## Example Visuals

The following images illustrate sample model predictions, patch extraction, and training results. Place the supplied image files in an `images/` folder at the repository root with the exact filenames shown below so they render correctly in this README.

- **Model sample prediction (truck):**
- <img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/faee0925-77d3-4849-a5c0-7ef58d8cd025" />

- **Image size and patch info:** 
- <img width="496" height="508" alt="Screenshot 2026-06-11 183456" src="https://github.com/user-attachments/assets/03f8d72b-88e2-4b88-91b6-8e7328aa04f9" />

- **Patch grid visualization:**
- <img width="329" height="328" alt="image" src="https://github.com/user-attachments/assets/7862153f-6220-44ea-b92d-e4959f4a52e1" />

- **Training/test results snapshot:**
- <img width="1040" height="68" alt="Screenshot 2026-06-11 183642" src="https://github.com/user-attachments/assets/b85b6f89-6e67-47e5-8e56-655a09d9c028" />


If you'd like, I can add these image files into the `images/` folder for you — tell me to proceed and I'll place them there and mark the todo list accordingly.

## How to run

1. Open `Vision Transformer (ViT).ipynb` in Jupyter Notebook or JupyterLab.
2. Execute cells sequentially from top to bottom.
3. If you want a quick preview, open `Vision Transformer (ViT).html` in a browser.
4. The training checkpoint files are stored under `tmp/checkpoint`.

## Notes

- This implementation uses a very small subset of CIFAR-10 for experimentation and faster iteration.
- The notebook code was verified to load CIFAR-10 via `keras.datasets.cifar10.load_data()`.
- If the project report mentions a different dataset, the code in the notebook is the source of truth for this repository.
- For larger-scale training, remove or expand the dataset slicing and adjust the image/patch sizes or model hyperparameters.

## Suggested improvements

- Add a `requirements.txt` for easier environment setup.
- Expand training to the full CIFAR-10 dataset.
- Add visualization of attention maps or training curves.
- Use a larger ViT variant or pre-trained weights for improved accuracy.
