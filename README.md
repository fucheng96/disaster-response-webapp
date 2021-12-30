# Disaster Response Categorizer Project

## Table of Contents

1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Program Execution](#Program-Execution)
4. [Web App Screenshots](#Web-App-Screenshots)
5. [Acknowledgements](#Acknowledgements)

## Overview

For this project, I was personally interestested in the 120-year Summer Olympics (or just 'Olympics') data from 1896 to 2016 to better understand:
- Had the Summer Olympics always been gender-balanced?
- How long did it take to make the transition from then male-dominated events?
- Which sports contributed to the rise of female sports events?
- Which countries first started to support their female athletes to compete in the Olympics?

There are 2 data files and 1 Jupyter notebook available here to showcase work related to the above questions. 
- Data/athlete_events.csv: Each row corresponds to an individual athlete competing in an individual Olympic event.
- Data/noc_regions.csv: Each row describes the name of the country based on the initials.
- SummerOlympicsDataAnalytics: Jupyter Notebook that performs the necessary analyses.

The notebook is exploratory in analyzing the data in relation to the questions above. Various sections were labelled using markdown cells in the notebook to help walkthrough thought process.

## Installation

1. Clone this git repository to your local workspace
   `git clone https://github.com/matteobonanomi/disaster-response.git`
   
2. Install following dependencies in addition to the standard libraries from Anaconda distribution of Python:
    - Natural Language Process Libraries - [NLTK](https://www.nltk.org/)
    - SQLlite Database Libraries - SQLalchemy
    - Web App and Data Visualization - Flask and Plotly

## Program Execution
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Web App Screenshots

1. It took around 120 years to reach gender balance. Crucial tipping points include the 1950s (afer World War II) and 1980s (after appointment of first female board members of the International Olympic Committee ('IOC'). 

![Screenshot 1]()

2. Athletics and Swimming are the key sports to the rise in women's participation in the Olympics, at both tipping points.

![Screenshot 2]()

3. Western countries such as UK, USA, Australia, Russia and France that sent the most female athletes around the 1950s and continue to do so in the 1980s till this date.

![Screenshot]()


## Acknowledgements

Credits to user named rgriffin from Kaggle that scraped the data from www.sports-reference.com. You may find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results).
