# INFO 521 - Final Project Log - CLK

###### tags: `INFO 521`

[TOC]

### Oct 06, 2022 (Thu.): reproducing [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)

#### 1. Install
- Create conda environment in `Steward`
    - `conda create --name TrackNetV2 python=3.5`
    - ~~no python 3.5.2 in conda, using python 3.5.6 instead~~
        - DEPRECATION: Python 3.5 reached the end of its life on September 13th, 2020. Please upgrade your Python as Python 3.5 is no longer maintained. pip 21.0 will drop support for Python 3.5 in January 2021. pip 21.0 will remove support for this functionality.
    - using python 3.6.13 instead of python 3.5 to suppot f-strings
- :x: install CUDA [(toturial)](https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) [(toturial2)](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)
    - install NVIDIA driver
        ```
        sudo apt-get install nvidia-384 nvidia-modprobe
        sudo add-apt-repository ppa:graphics-drivers/ppa
        ```
    - `http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /`
    - Not yet successfully instead CUDA, try to move on first
- Install python packages
    ```
    sudo apt-get install git
    sudo apt-get install python3-pip
    pip install pyqt5
    pip install pandas
    pip install PyMySQL
    pip install opencv-python
    pip install imutils
    pip install Pillow
    pip install piexif
    pip install -U scikit-learn
    pip install keras
    git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
    ```
    - add 
    ```
    pip install tensorflow
    pip install matplotlib
    ```
    

#### 2. Prediction for a single video

- Generate the predicted video and the predicted labeling csv file
```{py}
python3 predict.py --video_name=/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/DataSet/profession_dataset/match1/rally_video/1_01_00.mp4 --load_weights=model_33
```
- output a `1_01_00_predict.csv` file

---
### Oct 07, 2022 (Fri.): reproducing [TrackNet-Badminton-Tracking-tensorflow2](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2)

- Try to re-sun another repo that is based on TrackNetV2
    - `git clone https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2`
- Have similar error message that seems relative to the CPU/GPU things
- Need to go back to TrackNetV2

---
### Oct 13, 2022 (Thu.): error message when reproducing [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
- :eyes: error message 1 (solved?)
    - line 172 `y_pred = model.predict(unit, batch_size=BATCH_SIZE)`
    - `tensorflow.python.framework.errors_impl.InvalidArgumentError:  Default MaxPoolingOp only supports NHWC on device type CPU
	 [[node model_2/max_pooling2d_1/MaxPool (defined at predict.py:164) ]] [Op:__inference_predict_function_2866]`
    - Solutions [link](https://stackoverflow.com/questions/64666638/invalidargumenterror-default-maxpoolingop-only-supports-nhwc-on-device-type-cpu)
        - `NHWC` stands for `(n_samples, height, width, channels)`
        - you are using `(n_samples, channels, height, width)`, which Keras can't work with, at least on CPU.
        - add L169:`unit = np.transpose(unit, axes=[0,2,3,1])` before L172
            - `(1, 9, 288, 512)` -> `(1, 288, 512, 9)`
- :bomb: error message 2 (unsolved)
    - line 172 `y_pred = model.predict(unit, batch_size=BATCH_SIZE)`
    - `ValueError: Input 0 is incompatible with layer model_2: expected shape=(None, 9, 288, 512), found shape=(1, 288, 512, 9)`

---
### Oct 20, 2022 (Thu.): upload TrackNetv2 to github

- :coffee: error message 1 (solved):
    - when `git add TrackNetv2`
    ```
    hint: You've added another git repository inside your current repository.
    hint: Clones of the outer repository will not contain the contents of
    hint: the embedded repository and will not know how to obtain it.
    hint: If you meant to add a submodule, use:
    hint: 
    hint:   git submodule add <url> nodule
    hint: 
    hint: If you added this path by mistake, you can remove it from the
    hint: index with:
    hint: 
    hint:   git rm --cached nodule
    hint: 
    hint: See "git help submodule" for more information.
    ```
    - solution: [You have added a git repository inside another git repository](https://gist.github.com/claraj/e5563befe6c2fb108ad0efb6de47f265)
        - In TrackNetv2: `rm -rf .git`
        - In final-project-astrochialinko: `git rm -f --cached TrackNetv2`
        - In final-project-astrochialinko: `git add TrackNetv2`
- :coffee: errer message 2 (solved)
    - when `git push -u origin main`
    ```
    Enumerating objects: 27, done.
    Counting objects: 100% (27/27), done.
    Delta compression using up to 16 threads
    Compressing objects: 100% (26/26), done.
    Writing objects: 100% (26/26), 239.36 MiB | 8.08 MiB/s, done.
    Total 26 (delta 6), reused 0 (delta 0)
    remote: Resolving deltas: 100% (6/6), completed with 1 local object.
    remote: error: Trace: a150ba3fc920b48c23ff35e4bac6c9b180456f4b58e5397f07a19e43533fbbca
    remote: error: See http://git.io/iEPt8g for more information.
    remote: error: File TrackNetv2/3_in_3_out/model906_30 is 129.97 MB; this exceeds GitHub's file size limit of 100.00 MB
    remote: error: File TrackNetv2/3_in_1_out/model_33 is 129.97 MB; this exceeds GitHub's file size limit of 100.00 MB
    remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
    To github.com:ISTA421INFO521/final-project-astrochialinko.git
     ! [remote rejected] main -> main (pre-receive hook declined)
    error: failed to push some refs to 'git@github.com:ISTA421INFO521/final-project-astrochialinko.git'
    ```
    - solution: [Removing files from a repository's history](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
        - `git -rm --cached model_33`
        - `git rm --cached model906_30`
        - `git commit --amend -CHEAD`
        - `git push`

---
### Oct 26, 2022 (Wed.): trying to fix error message on Oct 13

- Jiacheng'r commit: 
    - The fix is to change each instance of `data_format='channels_first'` to `data_format='channels_last'` in file `3_in_1_out/TrackNet.py` and `3_in_3_out/TrackNet3.py`
    - This error message is associated with your fix for error message 1, where the `channel` dimension is moved from 2d to 4th position. And the `data_format` parameter is responsible for telling 
- try: typing `:%s/channels_first/channels_final` in `3_in_1_out/TrackNet.py`
- run: `python3 predict.py --video_name=/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/DataSet/profession_dataset/match1/rally_video/1_01_00.mp4 --load_weights=model_33`
- :bomb: error message 1 (unsolved)
    - same as error message 1 on Oct 13
        ```
        2022-10-26 13:26:57.710708: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
        2022-10-26 13:26:57.710727: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
        2022-10-26 13:26:58.672198: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
        2022-10-26 13:26:58.672229: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (dhcp132-143): /proc/driver/nvidia/version does not exist
        2022-10-26 13:26:58.672550: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
        To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
        Beginning predicting......
        2022-10-26 13:26:59.047538: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
        Traceback (most recent call last):
          File "predict.py", line 164, in <module>
            y_pred = model.predict(unit, batch_size=BATCH_SIZE)
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/engine/training.py", line 1751, in predict
            tmp_batch_outputs = self.predict_function(iterator)
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/eager/def_function.py", line 885, in __call__
            result = self._call(*args, **kwds)
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/eager/def_function.py", line 957, in _call
            filtered_flat_args, self._concrete_stateful_fn.captured_inputs)  # pylint: disable=protected-access
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/eager/function.py", line 1964, in _call_flat
            ctx, args, cancellation_manager=cancellation_manager))
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/eager/function.py", line 596, in call
            ctx=ctx)
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/eager/execute.py", line 60, in quick_execute
            inputs, attrs, num_outputs)
        tensorflow.python.framework.errors_impl.InvalidArgumentError:  Default MaxPoolingOp only supports NHWC on device type CPU
             [[node model_2/max_pooling2d_1/MaxPool (defined at predict.py:164) ]] [Op:__inference_predict_function_2866]

        Function call stack:
        predict_function
        ```

#### Note: Github comment
- create branck
    - git:(main) `git branch reproduce-clk`
    - git:(main) `git checkout reproduce-clk`
    - git:(reproduce-clk) `git add TrackNet.py`
    - git:(reproduce-clk) `git commit -m "change channels_first to channels_final in data_format"`
    - git:(reproduce-clk) `git push origin reproduce-clk`
- pull from main (for editing .md from website)
    - git:(reproduce-clk) `git checkout main`
    - git:(main) `git pull`
    - git:(main) `git log --oneline -5`
- merge branch from main
    - git:(main) `git checkout reproduce-clk`
    - git:(reproduce-clk) `git log --oneline -5`
    - git:(reproduce-clk) `git merge main`
    - git:(reproduce-clk) `git log --oneline -5`
    - git:(reproduce-clk) `git push origin reproduce-clk`
