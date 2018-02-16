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

""" Extract and save the audio from video files.
    Usage: $video2wav.py <input_files_path> <output_file_path
"""
import subprocess
import os
import sys

# Pre...
# textfile_path = 'absolute/path/to/videos.txt'
# Check if the path argument existed
if(len(sys.argv) < 3):
    print("Missing input or output path")
    sys.exit(0)
# Read the text file
# with open(textfile_path) as f:
    #content = f.readlines()
# you may also want to remove white space characters like n at the end of each line
# files_list = [x.strip() for x in content]

# Get list of video files from the input path
files_list = os.listdir(sys.argv[1])
# Extract audio from video.
# It already save the video file using the named defined by output_name.
for file_name in files_list:
    file_path_input = sys.argv[1] + file_name
    raw_file_name = os.path.basename(file_name).split('.')[0]
    file_path_output = sys.argv[2] + raw_file_name + '.wav'
    print('processing file: %s' % file_path_input)
    command = "ffmpeg -i " + file_path_input + " -codec:a pcm_s16le -ac 1 " + file_path_output
    subprocess.call(command, shell=True)
        #['ffmpeg', '-i', file_path_input, '-codec:a', 'pcm_s16le', '-ac', '1', file_path_output])
    print('file %s saved' % file_path_output)