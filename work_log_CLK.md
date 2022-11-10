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

- Jiacheng'r comment: 
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
---
### Oct 27, 2022 (Thu.): keep trying to fix error message on Oct 13

- Jiacheng'r suggestions
    - In `3_in_1_out/predict.py` file
        - L97 `# model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})`
        - L98 add `model = TrackNet(HEIGHT, WIDTH)`
        - L99 add `model.load_weights(load_weights)`
        - L166 add `unit = np.transpose(unit, axes=[0,2,3,1])`
    - In `3_in_1_out/TrackNet.py` file
        - type `:%s/channels_final/channels_last` 
- run: `python3 predict.py --video_name=/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/DataSet/profession_dataset/match1/rally_video/1_01_00.mp4 --load_weights=model_33`
- :eyes: error message 1 (solved?)
    - same as error message 1 on Oct 13
    ```
    Traceback (most recent call last):
      File "predict.py", line 98, in <module>
        model = TrackNet(HEIGHT, WIDTH)
      File "/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/final-project-astrochialinko/TrackNetv2/3_in_1_out/TrackNet.py", line 69, in TrackNet
        x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x3], axis=1)
        ...
    ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs shapes: [(None, 2, 72, 512), (None, 2, 72, 256)]
    ```
    - solution: `axix=1` -> `axix=3`
        - modify L71: `x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x3], axis=3)`
        - modify L89: `x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x2], axis=3)`
        - modify L102: `x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x1], axis=3)`
- :eyes: error message 2 (solved?)
    ```
    File "predict.py", line 98, in <module>
        model = TrackNet(HEIGHT, WIDTH)
      File "/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/final-project-astrochialinko/TrackNetv2/3_in_1_out/TrackNet.py", line 102, in TrackNet
        x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x1], axis=3)
        ...
    ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs shapes: [(None, 8, 288, 128), (None, 9, 288, 64)]
    ```
    - solution: change shape from `(9,h,w)` -> `(h,w,9)`
        - modify L8: `imgs_input = Input(shape=(input_height,input_width,9))`
- :eyes: error message 3 (solved?)
    ```
    File "predict.py", line 98, in <module>
        model = TrackNet(HEIGHT, WIDTH)
      File "/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/final-project-astrochialinko/TrackNetv2/3_in_1_out/TrackNet.py", line 131, in TrackNet
        output = (Reshape((OutputHeight, OutputWidth)))(x)
        ...
    ValueError: total size of new array must be unchanged, input_shape = [288, 512, 1], output_shape = [512, 1]
    ```
    - solution: changle `o_shape[2]` -> `o_shape[1]`; `o_shape[3]` -> `o_shape[2]`
        - modify L124: `OutputHeight = o_shape[1]`
        - modify L125: `OutputWidth = o_shape[2]`
- :bomb: error message 4 (unsolved)
    ```
    Traceback (most recent call last):
      File "predict.py", line 99, in <module>
        model.load_weights(load_weights)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/engine/training.py", line 2361, in load_weights
        hdf5_format.load_weights_from_hdf5_group(f, self.layers)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/saving/hdf5_format.py", line 713, in load_weights_from_hdf5_group
        backend.batch_set_value(weight_value_tuples)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/util/dispatch.py", line 206, in wrapper
        return target(*args, **kwargs)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/backend.py", line 3775, in batch_set_value
        x.assign(np.asarray(value, dtype=dtype_numpy(x)))
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/ops/resource_variable_ops.py", line 902, in assign
        (tensor_name, self._shape, value_tensor.shape))
    ValueError: Cannot assign to variable batch_normalization/gamma:0 due to variable shape (64,) and value shape (512,) are incompatible
    ```
    - ???
- Found that I used the newer version of keras and tensorflow
    - previous version
        ```
        keras==2.6.0
        tensorflow==2.6.2
        h5py==3.1.0
        ```
    - downgrade to
        ```
        keras==2.2.4
        tensorflow==1.13.1
        h5py==2.10.0
        ```
- :bomb: error message 4 (unsolved)
    - similar as the above error message 4 (so labeled as same number)
    ```
    Traceback (most recent call last):
  File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1659, in _create_c_op
    c_op = c_api.TF_FinishOperation(op_desc)
    tensorflow.python.framework.errors_impl.InvalidArgumentError: Dimension 0 in both shapes must be equal, but are 64 and 512. Shapes are [64] and [512]. for 'Assign_2' (op: 'Assign') with input shapes: [64], [512].
    During handling of the above exception, another exception occurred:
    Traceback (most recent call last):
      File "predict.py", line 99, in <module>
        model.load_weights(load_weights)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/engine/network.py", line 1166, in load_weights
        f, self.layers, reshape=reshape)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/engine/saving.py", line 1058, in load_weights_from_hdf5_group
        K.batch_set_value(weight_value_tuples)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 2465, in batch_set_value
        assign_op = x.assign(assign_placeholder)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/ops/variables.py", line 1762, in assign
        name=name)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/ops/state_ops.py", line 223, in assign
        validate_shape=validate_shape)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/ops/gen_state_ops.py", line 64, in assign
        use_locking=use_locking, name=name)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 788, in _apply_op_helper
        op_def=op_def)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py", line 507, in new_func
        return func(*args, **kwargs)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3300, in create_op
        op_def=op_def)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1823, in __init__
        control_input_ops)
      File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1662, in _create_c_op
        raise ValueError(str(e))
        ValueError: Dimension 0 in both shapes must be equal, but are 64 and 512. Shapes are [64] and [512]. for 'Assign_2' (op: 'Assign') with input shapes: [64], [512].
    ```
#### model.summary
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 288, 512, 9) 0
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 288, 512, 64) 5248        input_1[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 288, 512, 64) 0           conv2d[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 288, 512, 64) 256         activation[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 288, 512, 64) 36928       batch_normalization[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 288, 512, 64) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 288, 512, 64) 256         activation_1[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 144, 256, 64) 0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 144, 256, 128 73856       max_pooling2d[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 144, 256, 128 0           conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 144, 256, 128 512         activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 144, 256, 128 147584      batch_normalization_2[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 144, 256, 128 0           conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 144, 256, 128 512         activation_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 72, 128, 128) 0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 72, 128, 256) 295168      max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 72, 128, 256) 0           conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 72, 128, 256) 1024        activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 72, 128, 256) 590080      batch_normalization_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 72, 128, 256) 0           conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 72, 128, 256) 1024        activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 72, 128, 256) 590080      batch_normalization_5[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 72, 128, 256) 0           conv2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 72, 128, 256) 1024        activation_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 36, 64, 256)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 36, 64, 512)  1180160     max_pooling2d_2[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 36, 64, 512)  0           conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 36, 64, 512)  2048        activation_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 36, 64, 512)  2359808     batch_normalization_7[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 36, 64, 512)  0           conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 36, 64, 512)  2048        activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 36, 64, 512)  2359808     batch_normalization_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 36, 64, 512)  0           conv2d_9[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 36, 64, 512)  2048        activation_9[0][0]
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 72, 128, 512) 0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 72, 128, 768) 0           up_sampling2d[0][0]
                                                                 batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 72, 128, 256) 1769728     concatenate[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 72, 128, 256) 0           conv2d_10[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 72, 128, 256) 1024        activation_10[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 72, 128, 256) 590080      batch_normalization_10[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 72, 128, 256) 0           conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 72, 128, 256) 1024        activation_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 72, 128, 256) 590080      batch_normalization_11[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 72, 128, 256) 0           conv2d_12[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 72, 128, 256) 1024        activation_12[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 144, 256, 256 0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 144, 256, 384 0           up_sampling2d_1[0][0]
                                                                 batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 144, 256, 128 442496      concatenate_1[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 144, 256, 128 0           conv2d_13[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 144, 256, 128 512         activation_13[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 144, 256, 128 147584      batch_normalization_13[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 144, 256, 128 0           conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 144, 256, 128 512         activation_14[0][0]
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 288, 512, 128 0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 288, 512, 192 0           up_sampling2d_2[0][0]
                                                                 batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 288, 512, 64) 110656      concatenate_2[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 288, 512, 64) 0           conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 288, 512, 64) 256         activation_15[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 288, 512, 64) 36928       batch_normalization_15[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 288, 512, 64) 0           conv2d_16[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 288, 512, 64) 256         activation_16[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 288, 512, 1)  65          batch_normalization_16[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 288, 512, 1)  0           conv2d_17[0][0]
__________________________________________________________________________________________________
reshape (Reshape)               (None, 288, 512)     0           activation_17[0][0]
==================================================================================================
Total params: 11,341,697
Trainable params: 11,334,017
Non-trainable params: 7,680
```

---
### Oct 28, 2022 (Fri.): fix the error message on Oct 13

- Jiacheng'r suggestions
    - In `3_in_1_out/TrackNet.py` file
        - type `:%s/BatchNormalization()/BatchNormalization(axis=-2)`
- run: `python3 predict.py --video_name=/home/chia-linko/Workshop/Course/Fall2022/INFO521_MachineLearning/Final_Project/DataSet/profession_dataset/match1/rally_video/1_01_00.mp4 --load_weights=model_33`
- It works! :tada: 

#### model.summary
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 288, 512, 9)  0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 288, 512, 64) 5248        input_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 288, 512, 64) 0           conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 288, 512, 64) 2048        activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 288, 512, 64) 36928       batch_normalization_1[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 288, 512, 64) 0           conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 288, 512, 64) 2048        activation_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 144, 256, 64) 0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 144, 256, 128 73856       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 144, 256, 128 0           conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 144, 256, 128 1024        activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 144, 256, 128 147584      batch_normalization_3[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 144, 256, 128 0           conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 144, 256, 128 1024        activation_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 72, 128, 128) 0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 72, 128, 256) 295168      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 72, 128, 256) 0           conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 72, 128, 256) 512         activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 72, 128, 256) 590080      batch_normalization_5[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 72, 128, 256) 0           conv2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 72, 128, 256) 512         activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 72, 128, 256) 590080      batch_normalization_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 72, 128, 256) 0           conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 72, 128, 256) 512         activation_7[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 36, 64, 256)  0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 36, 64, 512)  1180160     max_pooling2d_3[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 36, 64, 512)  0           conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 36, 64, 512)  256         activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 36, 64, 512)  2359808     batch_normalization_8[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 36, 64, 512)  0           conv2d_9[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 36, 64, 512)  256         activation_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 36, 64, 512)  2359808     batch_normalization_9[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 36, 64, 512)  0           conv2d_10[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 36, 64, 512)  256         activation_10[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 72, 128, 512) 0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 72, 128, 768) 0           up_sampling2d_1[0][0]
                                                                 batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 72, 128, 256) 1769728     concatenate_1[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 72, 128, 256) 0           conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 72, 128, 256) 512         activation_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 72, 128, 256) 590080      batch_normalization_11[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 72, 128, 256) 0           conv2d_12[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 72, 128, 256) 512         activation_12[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 72, 128, 256) 590080      batch_normalization_12[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 72, 128, 256) 0           conv2d_13[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 72, 128, 256) 512         activation_13[0][0]
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 144, 256, 256 0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 144, 256, 384 0           up_sampling2d_2[0][0]
                                                                 batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 144, 256, 128 442496      concatenate_2[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 144, 256, 128 0           conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 144, 256, 128 1024        activation_14[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 144, 256, 128 147584      batch_normalization_14[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 144, 256, 128 0           conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 144, 256, 128 1024        activation_15[0][0]
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 288, 512, 128 0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 288, 512, 192 0           up_sampling2d_3[0][0]
                                                                 batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 288, 512, 64) 110656      concatenate_3[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 288, 512, 64) 0           conv2d_16[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 288, 512, 64) 2048        activation_16[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 288, 512, 64) 36928       batch_normalization_16[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 288, 512, 64) 0           conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 288, 512, 64) 2048        activation_17[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 288, 512, 1)  65          batch_normalization_17[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 288, 512, 1)  0           conv2d_18[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 288, 512)     0           activation_18[0][0]
==================================================================================================
Total params: 11,342,465
Trainable params: 11,334,401
Non-trainable params: 8,064
```
---
### Nov 07, 2022 (Mon.): extrame frames from videos
- extrame frames from videos
    - Made `video2frames.py` files to extrame `.png` frames from `.mp4` videos
    - python file is in `final-project-astrochialinko/TrackNetv2/video2frame`
- successfully extrame frames from videos
    - It takes ~40 mins for all the videos (~200) using my destop 
---

### Nov 08, 2022 (Tue.): run gen_data_rally.py 
- run `python gen_data_rally.py` in dir `TrackNetv2/3_in_1_out`
    - :coffee: error message 1 (solved):
        ```
        File "gen_data_rally.py", line 34, in <module>
            a = img_to_array(load_img(p))
          File "/home/chia-linko/miniconda3/envs/TrackNetV2/lib/python3.6/site-packages/keras_preprocessing/image/utils.py", line 113, in load_img
            with open(path, 'rb') as f:
        FileNotFoundError: [Errno 2] No such file or directory: 'match1/frame/1_01_00/1.png'
        ```
        - solution: modify path
            - In `python gen_data_rally.py` file
                - L32, add `parent_path = '../../../DataSet/profession_dataset/'`
                - L34, modify `p = os.path.join(parent_path, game_list[0], 'frame', '1_01_00', '1.png')`
    - :coffee: error 2 (solved, Nov 9):
        - error without error message
            - python codes create null `npy` folder 
            - It should create `x_data_1.npy` ... in `npy` foler
        - solution: modify path
            - In `python gen_data_rally.py` file
                - L47, modify `all_path = glob(os.path.join(parent_path, game, 'frame', '*'))`
                - L51, modify `train_path[i] = train_path[i][len(os.path.join(parent_path, game, 'frame')) + 1:]`
                - L53, modify `labelPath = os.path.join(parent_path, game, 'ball_trajectory', p + '_ball.csv')`
                - L60, modify `r = os.path.join(parent_path, game, 'frame', p)`
- Successfully run `python gen_data_rally.py` in dir `TrackNetv2/3_in_1_out`
    - It takes ~40 mins runing with my destop
    - It creates `npy` folder and generate 62 `x_data_.npy` and 62 `y_data_.npy` which are 165 GB
    - `OSError: Not enough free space to write 2701983744 bytes`
        - Total videos should be ~200
        - Not enough space for all the 200 videos, try using 62 video first and move on
    
---
### Nov 10, 2022 (Thu.): run train_TrackNet.py 
- run `train_TrackNet.py` in dir `TrackNetv2/3_in_1_out`
    - typing `python3 train_TrackNet.py --save_weights=../weights --dataDir=./npy --epochs=10 --tol=200`
    - :coffee: error message 1 (solved):
        ```
        File "train_TrackNet.py", line 166, in <module>
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        ValueError: Error when checking input: expected input_1 to have shape (288, 512, 9) but got array with shape (9, 288, 512)
        ```
    - solution: 
        - L166, add `x_train = np.transpose(x_train, axes=[0,2,3,1])`
        - L175, add `x_train = np.transpose(x_train, axes=[0,2,3,1])`
    - Susscufully run the `train_TrackNet.py`
        - not yet sure how large the models are
        - not yet sure how long take the training run, just try 10 epochs first
        - not yet sure how to choose the `tol`, first try 200
