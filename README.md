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
