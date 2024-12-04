git clone https://github.com/mdegans/nano_build_opencv
cd nano_build_opencv

jtop
# check info for cuda version

gedit build_opencv.sh

# CUDA_ARCH_BIN ; 8.7 isnt for jetson nano, remove it

# CUDNN version should match the one from jtop/info 

./build_opencv.sh 4.8.0


jtop
# check cuda version again in info

opencv_version
# should be 4.8.0


python3
# version 3.6.9
# >> import cv2
# >> print(cv2.__version__)
# if not 4.8.0, move to next steps


which opencv_version

cd /usr
sudo find -name opencv_version

./bin/opencv_version
./local/bin/opencv_version

echo $PATH

python3 
# >> import sys
# >> print('/n'.join(sys.path))

opencv_version -v

gedit .bashrc
export PYTHONPATH=/usr/local/lib/python3.6/site-package:$PYTHONPATH




sudo apt install git-lfs

git clone https://github.com/opencv/opencv_zoo
