# Final project progress report
### ISTA421/INFO521

-------

**Project:** Machine Learning in Badminton : Clasification of different badminton strokes during a match

**Names:**
- Chia-Lin Ko
- Yuan Jea Hew  

-------


## Instructions

You report should be a short summary of your project progress. This report is relevant to to us, your instructors, and probably to the rest of the class.

Please use this report as a chance to organize your thoughts about what you are trying to do with your project, and to get feedback on your ideas, and the approaches that you have tried so far.

## Submission

Please commit this file to your GitHub repo (progress.md) AND to D2L.


-------

### GitHub repo usage
_Describe the current structure of your repo, the number of commits, and the steps you have taken to ensure the reproducibility of your code_

> The current structure of our repo have two branches. One is the main branch and the other branch for recording the changes of the scripts for reproducing the TrackNetV2 model with CPU instead of GPU. The main branch has 24 commits and the reproduce branch has 34 commits

### Summarize your data
_Describe the characteristics of your data, any transformations that you have considered, or potential issues that you have faced (e.g. missing data)_

> Dataset descriptions :
> - We have video recordings of professional badminton matches provided by BWF Youtube channel 
-- Cut and labelled
> - We have our own recordings of professional badminton matches 
> Potential limitations and differences :
> - Badminton court color difference in professional matches and our own recorded matches
> - Badminton match recording angle difference in professional matches and our own recorded matches
> - Badminton match background difference in professional matches and our own recorded matches
> - Players might obscure shuttlecock during gameplay, thus making shuttlecock tracking/stroke classification be difficult
> - Stroke differences from professional players and amateur players like us

### Describe your initial analysis strategy
_What was your initial plan?_

> To classify badminton strokes performed by the players during a professional badminton match

### What you have tried so far?
_Describe your current implementation_

> Reproducing TrackNetV2 code and modyfied GPU use to CPU use to accomadate with our computers.

### What worked and what did not
_Describe the challenges that you have faced so far and outline a few take-homes from your experience on this project_

> Succefully tested with our own data but did not get a video output due to video dimensions compatibility.

### What you plan to do next...
_Please define explicit goals for each of the remaining weeks (before the presentation is due)_

> - Week 1: 
> Implement our own amateur data into the TrackNetV2 model and analyze the results. Get more good data with varying angles of gameplay. Identify correct dimensions for video input of TrackNetV2.
> 
> - Week 2: 
> Evaluate baseline model. Add newly gathered data with training set and evaluate performance via Confusion Matrix. Compare with baseline model. 
> 
> - Week 3:
>  Continue with week 2 task and wrao things up. Prepare for presentation and report.

### Author contributions
_Describe the contributions of each of the members to the current version of the project_


Student 1: Chia-Lin Ko
- [x] Development of question / hypothesis;
- [x] Data research: search for relevant data to contribute to question;
- [x] Literature review;
- [x] Analysis strategy;
- [x] Analysis code;
- [ ] Code review;
- [x] Work planning and organization;
- [x] Improving teamwork and collaboration;
- [x] Testing code and procedures;
- [x] Writing report.
- [ ] Additional comments:

Student 2: YuanJea Hew
- [X] Development of question / hypothesis;
- [ ] Data research: search for relevant data to contribute to question;
- [X] Literature review;
- [ ] Analysis strategy;
- [X] Analysis code;
- [ ] Code review;
- [X] Work planning and organization;
- [ ] Improving teamwork and collaboration;
- [X] Testing code and procedures;
- [X] Writing report.
- [ ] Additional comments:
