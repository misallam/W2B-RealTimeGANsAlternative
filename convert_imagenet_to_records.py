import tensorflow as tf
from glob import glob
import os
from scipy.misc import imread, imresize
import numpy as np

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_dataset(blurred_folder, enhanced_folder, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    blurred_paths = glob(os.path.join(blurred_folder, '*.png'))
    enhanced_paths = {os.path.basename(path): path for path in glob(os.path.join(enhanced_folder, '*.png'))}
    
    for blurred_path in blurred_paths:
        filename = os.path.basename(blurred_path)
        if filename in enhanced_paths:
            blurred_image = imread(blurred_path)
            blurred_image = imresize(blurred_image, (128, 128))
            blurred_image_raw = blurred_image.tostring()
            
            enhanced_image = imread(enhanced_paths[filename])
            enhanced_image = imresize(enhanced_image, (128, 128))
            enhanced_image_raw = enhanced_image.tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'blurred_image_raw': _bytes_feature(blurred_image_raw),
                'enhanced_image_raw': _bytes_feature(enhanced_image_raw)
            }))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    tf.app.run()

