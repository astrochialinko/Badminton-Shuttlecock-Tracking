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

