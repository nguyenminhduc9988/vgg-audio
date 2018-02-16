# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A simple demonstration of using VGGish in extracting feature.

WAV files (assumed to contain signed 16-bit PCM samples) are read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are print into csv format

Usage:
  # Run WAV files through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory. Write the 
  # embeddings to csv files stored in output path
  $ python vggish_extract_features.py --input_path /path/to/input/files

  # Run WAV files through the model and also write the embeddings to
  # csv files stored in output path. The model checkpoint and PCA parameters are 
  # explicitly passed in as well.
  $ python vggish_extract_features.py --input_path /path/to/input/files \
                                    --output_path /path/to/output/files \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_extract_features.py
"""

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os

flags = tf.app.flags

flags.DEFINE_string(
    'input_path', None,
    'Path to wav files. Should contain signed 16-bit PCM samples.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'output_path', "features/",
    'Path to files contains extracted embedding features.')

FLAGS = flags.FLAGS


def main(_):
  # We run the examples from audio files from the input path through the model.
  # If none is provided, we generate a synthetic input.
  if FLAGS.input_path:
    wav_files = os.listdir(FLAGS.input_path)
  else:
    # Write a WAV of a sine wav into an in-memory file object.
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    # Convert to signed 16-bit samples.
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_files[0] = six.BytesIO()
    wavfile.write(wav_files[0], sr, samples)
    wav_files[0].seek(0)
  examples_batch = [vggish_input.wavfile_to_examples(FLAGS.input_path + wav_file) for wav_file in wav_files]
  print("data sample",examples_batch[0])

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    for i in range(len(examples_batch)):
        print("Batch number: ", i)
        [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch[i]})
        #print("embedding_batch: ",embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        np.savetxt(FLAGS.output_path + wav_files[i][:-4] + ".csv", postprocessed_batch, fmt='%i', delimiter=",")

if __name__ == '__main__':
  tf.app.run()

# For loading the features values into np array
#
#   numpy.genfromtxt(<filename>, delimiter = ",") 
#
