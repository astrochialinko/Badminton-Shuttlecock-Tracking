# INFO 521 - Final Project Proposal

## Brief Info
- **Topic**: Machine Learning in Badminton : Clasification of different badminton strokes during a match 
- **Authors**: Chia-Lin Ko, Yuan Jea Hew

---
## Research topic
Background :
- To classify badminton strokes performed by the players during a professional badminton match  
  - Badminton Strokes : Smashes, Drops, Clears, Drives, Net Shots, Lifts etc 
- Player Detection during match 
- Provide statistical analysis of players gameplay 
- Model :
  - Input data -> Video recording of a badminton match
  - Output -> Model prediction on badmionton strokes of players throughout the match (can be expanded to match analysis)  

Why it is important ?
- Helps efficiently analyze athletes performance and provide statistical insights of the match 
- To help improve and identify weak points in their gameplay
- These type of research work are not prevalent in the sport of badminton 

Knowledge gap (What we know/don’t know about the topic)
- We don't know how well it can generalize into other video recordings outside of BWF footages

---
## Data Sets
A brief description of the data required :
  - Focusing on men and woman singles event in badminton tournaments
  - Manually cut, trim, and label video data
  - Labels of badminton players performing different kinds of strokes 
   
How would you get such data ?
- Download the dataset from the website: [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
- Video data to be extracted from [Badminton World Federation (BWF)](https://www.youtube.com/c/bwftv) Youtube channel 
- Tha data is labeled by the badminton coach and the professional badminton players.

---
## Plans
A list of questions & corresponding analysis tasks you plan to do :
- Analysis tasks we plan to do
    - Attempt to retrieve available public datasets to cut down the time spent on labelling data (done!)
    - Familiar with our dataset
    - Reproduce the results from the previous work (pretrained models) of [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
    - Improve the model performance
    - Apply this model to our own data set
- Questions to ask
    - Are we tested upon solely how well our model performs ? (Accuracy numbers, confusion matrix etc)
    - Any possible pretrained models?
    - Using models outside of the scope of our class ? (Computer vision, custom DL etc)
    - How well can the model be applied to our own recording videos ? 
    - Why does the model work well or not well ? 
    - Implementing a paper ?
    
