Problem 

2

Automatic Zoom
MindMetrix Home Assignment – Biomarker Data Scientist  
This is a home assignment to assess how you approach problems: especially, how you 
conduct , implement, and explain a data analytics task with a synthetic  data set  mimicking  
real - world data , while  focusing on reproducibility and a clean architecture of your choice.  
This assignment will be followed by a  technical interview. Please aim to spend no more 
than  4h  on  the assignment.   
Data Set  
The data set captures participants performing 10 trials of a relaxation exercis e. Each trial 
consists of: 5s baseline, 30s relaxation, 5s break. During th is relaxation exercise, several 
physiological signals are captured  through a VR - based eye - tracking system . Assume that 
light conditions are constant . 
The aim is  to establish a relationship between state markers/ metadata  of the participants 
and physiological markers of the time series.  
The data set consists of two  .csv files:  
(1)  Subjects.csv: participant metadata . 
(2)  Timeseries.csv: time series data of physiological signals .  
Assignment   
Your task is to  implement a reproducible preprocessing pipeline suitable for physiological 
data . Afterwards, explore and create meaningful inter -   or intra - participant features and 
their relationship with participant metadata. This can involve clustering of participants.  
Finally, use at least one exploratory unsupervised or dimensionality reduction method on 
your engineered features. Clearly state your hypotheses and justify your feature 
engineering decisions based on physiological reasoning.  
How would you prove that t he detected (bio)markers are sensitive?  
Focus areas  
• Feature design thinking & signal interpretation  
• Physiological reasoning  
• Reliable data analytics  
• Presentation/visualization of results  
• Code structure and clarity  
Optional  
Describe how you would connect the data pipeline to a backend API.  
Tech Guidelines  
• Use  Python  for implementation.  
• Keep it simple but clean, with consistent naming and structure.  
• Focus on demonstrating clarity, reliability, and maintainable architecture.  
• Show awareness of scalability and future integration with backend systems.  
• Write down your assumptions if something is left open or ambiguous.  
• S pecify any external libraries used and justify non - standard choices.  
Deliverables  
• Notebook , Git Repository or ZIP file  that  runs  end - to - end from the raw CSV files  
• Short report (max 1 page) covering your approach to the problem, explaining 
processing steps, model results and interpretations , as well as limitations and what 
you would do with more time.  Additionally,  quickly describe how you allocated the 
time . 
Interview Discussion  
We will talk about how you approach such problems, about the workflow,   and the result. 
You may also describe how this code could integrate into our backend and how you think 
this could scale and be updated as  more user data is collected. In addition, we will discuss 
how to handle code freezes and version control under a regulated process.  
 
Expected effort  
~3.5  hours implementation + 30 min documentation.   The focus is on structured thinking 
and sound reasoning, not on perfect models.  
Submission  
Send a GitHub link , Notebook, or   ZIP archive to  job@mindmetrix.ch   with subject:  “[Your 
Name] -  Assignment – Biomarker Data Scientist”  
 
 
 
 
 
 
 
 
 


1. Problem + context (pupil dia, autonomous nervous syste, anxiety)
2. Dataset + goals
3. Preprocessing + validation (masks, quality combination, trial quality score)
   1. Plot raw data for participant with mostly or all score 2 trials
   2. PLot data for reject participant or someone with low score
4. Feature groups per trial + rationale behind them
5. Feature aggregation and row per subject - end of preprocessing
6. Feature selection (multicollinear, iqr)
7. Correlation analysis
   1. STAI scores, both correlations
   2. Less correlation with error, quality
   3. Candidates
   4. Permutation analysis
8. PCA analysis
   1. Scree
   2. PCA 1 vs PCA2, conclusion
9. PCA vs STAI scores
10. PCA loadings and candidates
11. Robustness with metadata
12. 1 x 2 plot with results summary, with conclusions/interpretations
13. Summary and future work