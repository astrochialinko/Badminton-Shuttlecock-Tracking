# INFO 521 - Final Project Meeting Minutes

###### tags: `INFO 521`

[TOC]

---
### Oct. 14, 2022 (Fri.)
- **Time:** Oct. 14, 2022 during 11:00-11:30
- **Location:** [Zoom meeting](https://arizona.zoom.us/my/cromanpa)
- **Participants:** Chia-Lin Ko, Yean Jea Hew, Cristian Román-Palacios
- **Agenda**
    - Error message for final project
- **Discussion**
    - CLK, YJH
        - When re-run [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2), we have some error message seems relative to CPU/GPU things and/or python/system environment
            - For example, see CLK's work log [(Oct-13-2022)](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/work_log_CLK.md#oct-13-2022-thu-error-message-when-reproducing-tracknetv2)
    - CRP
        - Consider using [HPC](https://public.confluence.arizona.edu/display/UAHPC) for the GPU resource
            - NetID: `clko`, `yuanjea`
        - CRP will try to re-run the [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
- **Summary**
    - upload the [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2) to the github repo
    - Sent email when we finished it

---

### Oct. 07, 2022 (Fri.)
- **Time:** Oct. 07, 2022 during 12:30-14:00
- **Location:** N309, Steward Observatory
- **Participants:** Chia-Lin Ko, Yean Jea Hew
- **Milestone:** run the [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2) code with the [Shuttlecock Trajectory Dataset](https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ) 
- **Discussion**
    - CLK
        - environment: Ubuntu 20.04.3 LTS + unknow GPU
        - unsuccessfully install CUDA, so I try run the code without install CUDA
            - can run `predict.py`, but the output file (.csv and .mp4) are only having the frame but without prediceted value
            - get error message relative to CPU/GPU
        - Try runing other github rope [TrackNet-Badminton-Tracking-tensorflow2](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2) that is based on TrackNetV2
            - Have error message relative to CPU/GPU
        - It seems that I should try either install the CUDA and use GPU to run or modify the script to run with CPU only
        - I will try to install the CUDA in my mechine
            - `lshw -C display`
            - [LINUX X64 (AMD64/EM64T) DISPLAY DRIVER](https://www.nvidia.com/Download/driverResults.aspx/193095/en-us/)
    - YJH
        - environment: Windows + NVIDIA GPU
        - can not successfully run `predict.py` 
        - get error message relative to keras package(?)
- **Summary**
    - Schedule office hour at 11:00 Oct 14, 2022 [(Zoom link)](https://arizona.zoom.us/my/cromanpa)
    - CL will try to install CUDA
    - YJ wil try to run using python 3.6 and look into keras package
- **Next Meeting** 
    - Time: Oct. 17, 2022 at 3:30 pm
    - Location: ML classroom
    - Goal: summary and discuss (1) the meeting of office hour on Oct, 24 (2) any further work progress

---

### Nov. 16, 2022 (Wed.)
- **Time:** Nov. 16, 2022 during 16:00-17:00
- **Location:** N309, Steward Observatory
- **Participants:** Chia-Lin Ko, Yean Jea Hew
- **Discussion**: Work progress and schedule
    - CL
        - Not sure if we need to do the classification, we can either (1) keep going for classifying the strokes as our initial goal or (2) try to do some analysis based on the trajectory data process
        - Our main goal is to let computer helps us analysis the video and do some statistics/analysis. By doing so, we can know how we mostly lose the scores and how we can improve  based on the data that produced by computers.
        - [TrackNetV2](https://github.com/ISTA421INFO521/final-project-astrochialinko/tree/main/TrackNetv2) github repo that (1) provided the datasat and (2) we successfully reproduced only helps us to do the shuttle tracking. To do the classification, we need to find other github repo
    - YJ
        - I'm open in doing that, although I am not sure how we can statistically analyze the shuttle tracking
    - CL
        - We have the coordinate of the birdies and we can get the coordinate of court (lines), so that we can get (1) the number of the rally (2) which locations (front, end, left, right) we miss the most, and maybe more…(?)
        - Although (most?) of the video we recorded does not includes all the court (lines) due to that the view of the camera is not wide enough
    - Tentative Schedule
        - Nov. 16 (Wed.) - Nov. 21 (Mon.): 
            - Select and labeled the test dataset video
            - Run the  [`accuracy.py`](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/TrackNetv2/3_in_1_out/accuracy.py) script
            - Find other github for the strokes classification
        - Nov. 21 (Mon.) - Nov. 23 (Wed.) (Tentative)
            - If need to do classification
                - reproducing the github repo
            - If only do trajectory detection
                - write script to analyze the trajectory
        - Nov. 28 (Mon.) - Nov. 30 (Wed.) (Tentative)
            - If need to do classification
                - write script to analyze the classification
            - If only do trajectory detection
                - keep writing script to analyze the trajectory
        - Nov. 30 (Wed.) - Dec 5 (Mon.)
            - Deal with final analysis if needed
            - Work on the Presentation and the final project paper
        - Dec 5 (Mon.): Presentation Day
- **Summay**
    - Schedule office hour with Cristian at 10:45 Nov 21, 2022 [(Zoom link)](https://arizona.zoom.us/j/8195763300) to check if we need to do the classification, or trajectory detection is enough.
    - Schedule office hour with Jiacheng at 15:30 Nov 22, 2022 [(Zoom link)](https://arizona.zoom.us/j/88476759965) for techinal/code parts suggestions.
    - To-do-list before next Monday
        - Select 2 videos from two different angels: one from top-down angle view and one from side view
            - YJ selects 2 MS games
            - CL selects 2 WS games
        - Label the videos using [Shuttlecock Labeling Tool](https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ)
        - Run the [`accuracy.py`](https://github.com/ISTA421INFO521/final-project-astrochialinko/blob/main/TrackNetv2/3_in_1_out/accuracy.py) to [provide the performance information](https://github.com/ISTA421INFO521/final-project-astrochialinko/tree/main/TrackNetv2#notebook_with_decorative_cover-5-provide-the-performance-information)
        - Try to find other GitHub repo for the strokes classification

