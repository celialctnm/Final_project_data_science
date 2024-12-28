# Music Emotion Recognition ♪ ♫

This project was carried out as part of the course **Introduction to Data Science** (NCU).

## Overview

The connection between music and emotion is a fascinating topic that interests musicians, psychologists,
and researchers. Various musical elements, such as chord progression, tonality, intensity, and tempo,
play important roles in shaping how we feel when we listen to music. For instance, certain chord
progressions can make us feel happy or sad, while choosing a major or minor tonality can change the
emotional tone of a piece. Intensity, which refers to how loud or soft music is, can create feelings of
excitement or calmness, and the beats per minute (BPM) can influence how urgent or relaxing a song
feels.

This exploration of music’s emotional language highlights its power to connect with our feelings,
making it an interesting area to study. As musicians, we are excited about this project. We want to
understand how artists share emotions through music using specific numerical parameters and check
whether these theories can be scientifically validated.

Our goal is to detect the overall mood of a song by examining different factors like chord progression,
intensity, tonality... By understanding how these musical elements influence listeners' emotions, we hope
to identify the main feelings that artists want to convey. This research will deepen our appreciation of
music and its emotional impact.


## Architecture

- **data/**
    - **custom_dataset/custom_data.csv** : The first dataset, which we have customized as best as possible to address our problem statement. 
    - **final_dataset/data_moods.csv** : The final dataset we selected, the one that provides us with the best input data.
    - **graph**/ : The saved plots from main.py.
- **src/**
  - **component.py** : This file contains functions for visualizing plots and results (confusion matrix, classification report, etc.).
  - **custom_dataset.py** : The file to execute to access the results and tests of our various models with our customized dataset (these are not the best results).
  - **main.py** : The main file to execute, containing the final and optimized implementation of our three models (SVM, Random Forest, KNN) based on the Kaggle dataset.

## Setup & run

### Python Version and librairies

- Python 3.9
- Pandas
- Matplotlib
- Scikit-Learn
- Seaborn

### Execute the program

Run the file main.py to execute the version using the dataset **Spotify Music Data to Identify the Moods.**

You can also run custom_dataset.py to obtain the detailed results described in the report for the **customized dataset**.

## References

**Spotify Music data to identify the moods** https://www.kaggle.com/datasets/musicblogger/spotify-music-data-to-identify-the-moods 

