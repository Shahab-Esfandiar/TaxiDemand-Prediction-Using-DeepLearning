# Manhattan Taxi Demand Prediction Using Deep learning
![manhattan_taxis_image](https://github.com/user-attachments/assets/d03a2d17-d307-494a-a8ef-c49420059cff)

*This is my __final thesis__ project for __Bacherlor's KNTU Geomatic engineering__.*

* [1. Introduction](#1_Introduction)
* [2. What Data Have I used?](#2_What-Data-Have-I-used)
* [3. Methodology & Internal Structure](#3_Methodology-And-Internal-Structure)
   * [3.1. Data Prepration](#3_1_Data-Prepration)
   * [3.2. Exploratory Data Analysis(EDA)](#3_2_Exploratory-Data-Analysis(EDA))
   * [3.3. Data preprocessing](#3_3_Data-Preprocessing)
* [4. Model design](#4_Model-Design)
* [5. Evaluation](#5_Evaluation)
* [6. Execution Guide](#6_Execution-Guide)
* [7. Conclusions](#7_Conclusions)

---
# 1_Introduction
*__Taxi Demand Prediction__ is a Deep learning application designed to forecast the number of taxi requests in specefic region. in this case, I choose __Manhattan__ in New York and predict demands for the __next week__. The predictions are displayed by city zone and broken down into __hourly__ intervals.*

*So I aimed for my project to be __profitable in the real world__. If a taxi driver could accurately predict which boroughs or areas will have the highest demand, he could optimize his workday by focusing on those areas. This would allow him to either earn more money in the same amount of time or save time for his family and personal life, ultimately improving his quality of life.*

- ***The purpose of this document is to provide a brief overview of the project.***

---

# 2_What Data Have I used?
- **2020-2022 Yellow Taxis data** : *I downloaded this dataset from [NYC Open Data](https://opendata.cityofnewyork.us/), a free public data source of __New York__ City.* 
*The dataset includes 17 fields such as `Pick-up time and date`,`Pick-up location`, `Drop-off location`, `Fare amount`, `Payment method` ..etc. you can have a look and review the full features description from the [data dictionary here](https://data.cityofnewyork.us/api/views/2upf-qytp/files/4a7a18af-bfc8-43d1-8a2e-faa503f75eb5?download=true&filename=data_dictionary_trip_records_yellow.pdf).*

- **Weather history dataset** : *I wanted to include some weather parameters such as `temp` and `precip` data to train the model, as it is sensible to think that rain would affect the taxi demand. I used __Web Scrapping__ and downloaded it from the [visual crossing website](https://www.visualcrossing.com/) for the period between Jan-2020 to Apr-2022. but it takes a couple of days for them to respond your query so you can also download it from my repo in the Data folder.*

- **Polygon shape file** : *In order to visualize the results I needed geometric data. This `.shp` file represents the boundaries zones for taxi pickups as delimited by the New York City __Taxi and Limousine Commission (TLC)__. You can download the file from several [websites](https://archive.nyu.edu/handle/2451/36743) or my Repo.*
  
![taxi_zone_map_manhattan](https://github.com/user-attachments/assets/6fee1df0-5d13-46f9-98ec-009d309a1518)

---

# 3_Methodology And Internal Structure
*I can divide the project into 3 main parts: Data Prepration & Processing, Model Designing & Tuning and Evaluation & Visualization. However, the roadmap below outlines the project steps in more detail.*

![Flowchart](https://github.com/user-attachments/assets/41ac6537-9f49-4c97-9689-fdfae618dba0)

## 3_1_Data-Prepration
*In this section, the data cleaning steps were carried out, which include :*

![Screenshot 2024-07-24 140000](https://github.com/user-attachments/assets/9ac7c11b-e1a7-4bbd-bd2b-04dbca4afbe0)

1. Removing outliers and null data
2. Deleting data outside the spatial and temporal limits of the study area.
3. Checking the wrong data based on research and information of the study area and removing them
---

*After performing the above steps and based on the graph obtained, more than 90% of taxi requests are related to the Manhattan area.*

![PUNY_req](https://github.com/user-attachments/assets/f62cdda3-9a97-4154-b6fc-9c0ccd2e7477)

*The features that were used in the second & third step to unify the dataset were `Trip Distance`, `Passenger count`, `Fare amount` and `Location ID`. So, out of a total of 80 million records, 64 million records remained for use in this project. Also, by aggregating the requests, we converted the dataset into one-hour intervals using `Datetime` column.*

---

## 3_2_Exploratory Data Analysis(EDA)
*In this section, We analyzed the data statistically and determine the appropriate columns based on the graphs obtained. The analyzes that have been carried out include :*

**1. Map pickups by zone :** *I plotted a choropleth map showing Manhattan taxi zones by number of pickups, highlighting the top ten in red.*

![choropleth](https://github.com/user-attachments/assets/f183a9a2-49b9-4c3e-8d1c-954d43afd1a9)
---

**2. Linear chart pickups over time :** *I analysed pickups' evolution over different periods of time looking for patterns.*

  - **Pickups evolution over Months**

    ![Month](https://github.com/user-attachments/assets/ce5244c7-2098-4fc2-a629-f3262b9c73fe)
    ---

  - **Pickups evolution over Days**

    ![Day](https://github.com/user-attachments/assets/c35ea9c6-87af-452a-a5ee-a2bf60d16034)
    ---

  - **Pickups evolution over Hours**

    ![Hour](https://github.com/user-attachments/assets/5234d83f-1ad1-4fb4-84f8-0780cf112ad5)
    ---

**3. Pairwise Relationships** *As the number of pickups or **total demand** is very stable over time, I analysed only one month (so that my system can handle the it).
These are the relations found between the variables :*
- **Total demand - weekend :** *There are more pickups during the weekend.*
- **Total demand - weekday :** *There are more pickups on Saturday, Friday, Thursday, in this order. This variable is related to ``weekend`` but it contains more granularity about pickups distribution so I will keep this variable and remove ``weekend`` when training the models.*
- **Total demand - hour :** *There are more pickups between 23:00 and 3:00. This could be because there is not public transport.*
- **Total demand - day :** *There is a clear weekly pattern so this information is already given by ``weekday``. Therefore, I will not use ``day``to train the models.*

![Pairwise](https://github.com/user-attachments/assets/5287b6e8-c233-474e-b1c3-5bc29b995b35)
---

*In the end, The below treemap shows the top zones in terms of taxi pick-up count. We will develop Deep Neural Network model called **CNN-LSTM Encoder-Decoder** with `Attention mechanism` to forecast the pick-up trips in Manhattan's zones.*

![Heatmap](https://github.com/user-attachments/assets/5a584e08-fcad-4709-a3d1-5d17df8c1a36)

---

## 3_3_Data preprocessing
*In this section, we perform the necessary processing suitable features and data convert into the format that the input of the model should have.*
- **Feature engineering**
  - *First, we remove irrelevant features so that the model is properly trained.*
  
    ![Screenshot 2024-07-24 163519](https://github.com/user-attachments/assets/a91477dd-9020-4e8e-99c8-85d2cab4102d)
    ---

  - *Next, separate the passenger column to get the total number of passengers for each __1-hour__ period and and finally add the passengers column as population to our taxi data.*
    
  - **Spatial-Time based parameters :**
  *The main part of the feature engineering is related to these parameters, so that the steps to create it are done as follows*
    1. Creating year, month, day and hour columns based on the existing datetime
    2. Add weekday, weekend and holiday columns based on the first column
    3. Creating columns related to each ID location and summing up the number of its requests in each desired time period

  ![Screenshot 2024-07-24 165305](https://github.com/user-attachments/assets/be060cbb-e137-421e-9ea0-367edff06ac3)
  ---

  - **Weather parameters :**
    *In this section, we enter the data related to __New York's Central Park meteorological stations__ obtained through web scraping and perform the necessary processing such as the specific time frame of the project and other things. For this project, only `temperature`, `cloudcover` and `precipitation` can be suitable features*

    ![Screenshot 2024-07-24 170626](https://github.com/user-attachments/assets/0aa76b6b-6b3b-4dbb-8a01-ff0c8cf86ad0)
    ---

- **Feature Extraction :**
    *First, We can use Correlation diagram to understand the relationships between features to remove or change some features if necessary.*

     ![Correlation](https://github.com/user-attachments/assets/210a332f-1856-4818-935d-f61db653a1e4)
    ---

  *By analyzing the results of these two graphs and the previous statistical comparisons, as expected, we can find out :*
    1. the characteristics of the `population`, `hour` and `month` had the greatest impact.
    2. the situations of cloudy and official holidays due to their small amount does not have much relationship with the number of taxi requests throughout the `year`.
    3. So we can ignore the characteristics of the day, since the days of the week have a greater impact, and also remove the parameters of `rainfall` and `holidays`.
 
*Then, we delete the extra features from the dataset and sort the remaining columns so that the taxi demand columns are at the end of the dataset and normalized that. Then we changed the data format to `time series` mode and saved them for 3 categories of __train, validation and test__. So the final datset for the model like this :*

  ![Screenshot 2024-07-24 171644](https://github.com/user-attachments/assets/b6aab2db-dd49-41e5-a44b-361434f370dc)
  
---

# 4_Model Design
*In this step, we design the neural network model using **CNN-LSTM layers** and combined that with **Multihead Attention mechanism**.*

- *for the basic model, we determine the default parameters such as batch size and `Adam's` optimization and compile it. Then we run it with **100 epochs** on our training and validation data to obtain the initial accuracy of the architecture.*

  ![opt_model_diagram](https://github.com/user-attachments/assets/4e9e0794-d513-4032-bb55-dd27b2876f8e)
  ---
  
- **Hyperparameter Tuning :** *In this optimization, the following items are checked using `RandomSeachCV`.*
  1. The number of `convolutional` layer filters
  2. Amount of random removal of `neurons`
  3. The number of neurons in the `LSTM` layers
  4. LSTM layer `dropout` value
  5. The number of neurons in hidden or `dense` layers
  6. Output layer `activation` function
  7. Model `optimizer` for compilation

  ![Screenshot 2024-07-24 174948](https://github.com/user-attachments/assets/66c3ccc9-9776-4296-9365-5b7f9b23ad75)
  ---

# 5_Evaluation
*In this step, we save the final __trained model__, which has a better __accuracy__ than the initial model, and display its __Loss__ and `RMSE` charts. Then we load the that and run on our __test__ dataset to get the real accuracy of the model.*

![Training_rmse_opt](https://github.com/user-attachments/assets/c481cae6-11f2-445d-bd3c-a2778eda70e5)

---

- **Metrices :** *in this project, the criteria are MAE for loss, RMSE and  MSLE usage. The prediction accuracy of the model on the test data is available in the table below.*
  
  -|MAE|RMSE|MSLE
  -|-|-|-
  Base Model|11.2077|20.3724|0.5004
  Tuned Model|9.7939|327.5211|0.1845

- *As a visual comparison, for a random one-hour periods and 30 regions with different demand distributions, we display the graph of actual and predicted values.*

  ![Time_2022-06-18_18](https://github.com/user-attachments/assets/2b7241ee-568e-4fca-9235-bbb35203a8bb)
  ---

  *Based on the graphs, it can be seen that the model has understood the request pattern well, but in some areas, due to the lack of data, the model's prediction accuracy has decreased.*

---

# 6_Execution Guide
### If you want to run the WebApp

  1. Clone or download the **repository**.
  2. navigate to ``TaxiDemand-Prediction-Using-DeepLearning\``.
  3. Type ``streamlit run StreamlitMap.py`` in the command line.
  4. Copy the returned Network URL like `(http://172.19.0.1:8501)` and paste in your internet browser.
  5. That´s it! The app takes a couple of seconds to load cause using __Big data__.

**Note**:  *There are multiple environments on which you can execute the app and I am not capable to cover them all. So these steps refer to my personal environment(Windows 11)*

  ![choropleth_combined](https://github.com/user-attachments/assets/fd129f61-f176-4fd0-ac64-8dbd72e2e5a5)
  
---

# 7_Conclusions
*As expected, the accuracy of the model is much better in the regions where the number of requests is __higher__. According to all the available data and results, it can be said that the model is accurate. However, it is possible to increase the accuracy of the model in __low demand__ areas by increasing the amount of data or their time intervals.*

*Also, to increase the accuracy of the model in general, data such as traffic and the amount of taxi drop-offs in each area can be used as effective features.*

*It seems like there is a big gap between the product creation and the product use. There are lots of tools for data scientist to analyse data, clean, transform, train models, visualize data, etc. But once all that work is done, we need to put into production, create a product that someone unskilled in the field can use, for example, a web application. Streamlit seems to be the best option, and yet it is in very early stages. For this reason I encountered a significant number of bugs in streamlit while trying to integrate an altair choropleth map. This made me realise of how young is still the Data Science field and some of its tools.*

---

**I hope this repo is useful for you and I will be honored if you share your thoughts about the project with me 😄.**






