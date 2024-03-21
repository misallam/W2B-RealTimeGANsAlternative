# W2B-RealTimeGANsAlternative

W2B-RealTimeGANsAlternative offers a cutting-edge solution for real-time image processing, leveraging an alternative approach to traditional GANs for enhanced, efficient image enhancements.

Model Training
===============

Before using this code, it is necessary to create the dataset, using
convert_imagenet_to_records.py. It will need some filepaths edited
for your system.

To train an imagenet model, run "bash train_imagenet.sh"
Having TensorFlow0.8 is recommended.

Inception Score
===============

model.py provides a code to compute Inception Score. 
Inception model is downloaded automatically.
The function to compute score is called "get_inception_score"

Training Steps
==============

To train on your custom dataset of blurred and enhanced images using the modified scripts, follow these steps. Ensure you have TensorFlow installed and your dataset prepared in the specified structure.

### 1. Prepare Your Dataset

- Make sure your dataset is organized into two folders: one for blurred images (`original`) and one for enhanced images (`enhanced`). Each image in the `original` folder should have a corresponding image with the same filename in the `enhanced` folder.

### 2. Convert Your Dataset to TFRecords

- Modify the paths in the updated `convert_imagenet_to_records.py` script to point to your dataset's folders.
  - Set `blurred_folder` to the path of your blurred images folder.
  - Set `enhanced_folder` to the path of your enhanced images folder.
  - Set `output_file` to the desired output path for the TFRecords file.
- Run the modified `convert_imagenet_to_records.py` to generate a TFRecords file of your dataset:
  ```bash
  python convert_imagenet_to_records.py
  ```

### 3. Train the Model

Before running the training script, ensure the model (`DCGAN` in `model.py`) and training script (`train_imagenet.py`) are appropriately configured for your task. This includes adjusting the model architecture, input preprocessing, and any other parameters specific to your images or training objectives.

- Adjust any model-specific parameters or configurations in `train_imagenet.py` or the model definition itself to suit your needs. This might involve modifying the network architecture, loss functions, or training procedure based on whether you're doing image enhancement, super-resolution, etc.

- Modify the `train_imagenet.sh` script (if using) or directly set the appropriate flags in `train_imagenet.py` for training parameters, such as:
  - `--dataset` to the name of your dataset (or use the path to the TFRecords file if your training script is adjusted to accept it directly).
  - `--is_train` set to `True` to enable training mode.
  - `--checkpoint_dir` to specify where to save model checkpoints.
  - `--sample_dir` to specify where to save sample outputs during training.
  - Update any other paths or parameters as needed, like image sizes or batch sizes, to match your dataset's characteristics.

- Run the training script. If using the shell script:
  ```bash
  bash train_imagenet.sh
  ```
  Or, if running `train_imagenet.py` directly, ensure all necessary flags and parameters are set accordingly, either within the script or as command-line arguments.

### 4. Monitor Training Progress

- Keep an eye on the training process through logs, sample outputs, or any visualization tools you've integrated. Adjustments may be needed based on the observed performance, such as tuning hyperparameters or modifying the model architecture.

### Note

Training a GAN, especially on tasks like image enhancement, can be challenging and may require careful tuning of the model and training parameters. Experiment with different architectures, loss functions, and training strategies to achieve the best results for your specific task.
