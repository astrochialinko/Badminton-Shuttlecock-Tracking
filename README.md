# Badminton Shuttlecock Tracking

###### tags: `INFO 521`

Badminton Shuttlecock Tracking using Convolutional Neural Networks

- INFO 521 Final projects (Fall 2022)
- Authors: Chia-Lin Ko, Yuan Jea Hew

## Video Presentation

[![Watch the video](https://img.youtube.com/vi/4aSbkJG9M0o/maxresdefault.jpg)](https://youtu.be/4aSbkJG9M0o)


## Dataset

[Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
- Badminton games
    - 15 professional Men singles (MS) and Women singles (WS)   
    - 3 amateur games (only MS) 
    - Total frames : 55563 (46038 professional / 9525 amateur) 
- Various viewing angles 
- Red and Green badminton court colors
- Various background conditions

[Our Self Recorded Dataset](https://drive.google.com/drive/folders/1mNntqLRaQkIhUmLZPC19Yc7HXesgBrpQ?usp=share_link)
- Badminton games 
    - 2 amateur Men singles (MS) and 2 amateur Women singles (WS) 
    - Total frames: 2098 amateur
- Various viewing angles (side and top)
- Wooden badminton court colors
- Various background conditions


[Shuttlecock Labeling Tool](https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ)
- Generate `.csv` and `.npy` file for to (1) estimate performance and (2) train the model (future work)
- Labelling Pipeline
    1. Resize video clips to 1280x720p
    2. Convert video to image sequence (frames)
    3. Label frames using labelling tool
    4. Output .csv file and .npy file
- It is super Time Consuming!!! (require lots of manpower)
    - 15 sec video (450 frames) for 1 hour!

## File Structure

```     
TrackNetv2
    ├── 3_in_1_out/
    │     ├── predict.py
    │     ├── accuracy.py
    │     ├── TrackNet.py
    │     ├── …
    │     └── model_33 
    ├── 3_in_3_out/
    │     ├── predict3.py
    │     ├── accuracy3.py
    │     ├── TrackNet3.py
    │     ├── …
    │     └── model906_30 
    └── video2frame/
          └── video2frames.py
```        
- This project uses **TrackNet Model** to track the shuttlecock with scripts in `3_in_1_out` folder
- The `model_33` and `model906_30` are removed from the repository because they GitHub's file size limit of 100 MB
- Those two model can be downloaded from the original [TrackNetv2 GitLab Repo](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
- The scripts of TrackNetV2 Model are in `3_in_3_out` folder (future work)
- The scripts of `predict.py`, `accuracy.py`, and `TrackNet.py` have been modified so that we can run the scripts without GPU (e.g., CPU only) to accomadate with our computers.

## Setup Instructions

1. Install Python 3.6.13
2. Create a new virtual environment (optional)
3. Install Python packages
    ```
    pip install -r requirements.txt
    ```
#### Disclaimer 
This project runs on certain versions of Python and conda environment. We cannot guarantee it will work with other versions.
- Python 3.6.13
- conda 4.12.0


## Prediction
```
python3 predict.py --video_name=<videoPath> --load_weights=<weightPath>
```

For example:
- `<videoPath> = ../DataSet/OurOwnRecording/chialin_neha.mp4`
- `<weightPath> = model_33`

Then, the command will be `python3 predict.py --video_name=../DataSet/OurOwnRecording/chialin_neha.mp4 --load_weights=model_33`

The outputs are `predict.mp4` and `predict.csv` files

The `predict.mp4` looks like

[![Watch the video](https://img.youtube.com/vi/vjk_LwsQJ3c/maxresdefault.jpg)](https://youtu.be/vjk_LwsQJ3c)

The `predict.csv` files includes below information

- **Frame**: represent the frame of video
- **Visibility**: the shuttlecock is visible or not at this frame
    - 0 : invisible
    - 1 : visible
- **X, Y**: the coordinate of shuttlecock
    - If shuttlecock is invisible now, then X, Y is 0
- **Time**: the time of the video


## Estimate the performance

Estimate the performance by running the command below  
```
python3 accuracy.py --load_weights=<weightPath> --dataDir=<npyDataDirectory> --tol=<toleranceValue>
```
- Hint: You may need to generate the `.npy` file first for your own data using the [Shuttlecock Labeling Tool](https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ) mentioned above

For example:
- `<weightPath> = model_33`
- `<npyDataDirectory> = npy`
- `<toleranceValue> = 4`

Then, the command will be `python3 accuracy.py --load_weights=model_33 --dataDir=npy --tol=4`

The output looks like
```
Beginning evaluating......
==========================================================
Finish evaluating data1:(TP, TN, FP1, FP2, FN)=(191, 38, 78, 91, 5)
==========================================================
Number of true positive: 191
Number of true negative: 38
Number of false positive FP1: 78
Number of false positive FP2: 91
Number of false negative: 5
accuracy: 0.5682382133995038
precision: 0.5305555555555556
recall: 0.9744897959183674
Done......
```

## Notes

- [proposal.md](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/proposal.md): description of the initial idea of this project, made on Sep. 30.
- [progress.md](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/progress.md): progress report, made on Nov. 9.
- [Meeting_Minutes.md](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/Meeting_Minutes.md): meeting minutes.
- [work_log_CLK.md](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/work_log_CLK.md): description of the work log. Mostly are the error messages we have and how we solve them.




## References

- TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sport Applications [(IEEE Conference Preceedings)](https://www.computer.org/csdl/proceedings-article/avss/2019/08909871/1febOkjOevC)
- TrackNetV2: Efficient Shuttlecock Tracking Network [(IEEE Conference Preceedings)](https://ieeexplore.ieee.org/document/9302757)
- TrackNet [(GitLab)](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet)
- TrackNetV2 [(GitLab)](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)

## Acknowledgements

We would like to acknowledge support for Prof. Cristian Román-Palacios and Jiacheng Zhang for their guidance and support throughout the semester.
