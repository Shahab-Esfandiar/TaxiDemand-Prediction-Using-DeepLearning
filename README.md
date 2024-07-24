# Manhattan Taxi Demand Prediction Using Deep learning
![manhattan_taxis_image](https://github.com/user-attachments/assets/d03a2d17-d307-494a-a8ef-c49420059cff)

*This is my __final thesis__ project for __Bacherlor's KNTU Geomatic engineering__.*

* [1. Introduction](#1_Introduction)
* [2. What Data Have I used?](#2_What-Data-Have-I-used)
* [3. Methodology & Internal Structure](#3.Methodology-Internal-Structure)
   * [3.1. Data Prepration](#2.1.Data-Prepration)
   * [3.2. Exploratory Data Analysis(EDA)](#2.2.Exploratory-Data-Analysis(EDA))
   * [3.3. Data preprocessing](#2.3.Data-Preprocessing)
   * [3.4. Data Generator](#2.4.Data-Generator)
   * [3.5. Model design](#2.5.Model-Design)
* [4. Evaluation](#4_Evaluation)
* [5. Execution Guide](#5.Execution-Guide)
* [6. Conclusions](#6.Conclusions)

---
# 1_Introduction
*__Taxi Demand Prediction__ is a Deep learning application designed to forecast the number of taxi requests in specefic region. in this case, I choose __Manhattan__ in New York and predict demands for the __next week__. The predictions are displayed by city zone and broken down into __hourly__ intervals.*

*So I aimed for my project to be __profitable in the real world__. If a taxi driver could accurately predict which boroughs or areas will have the highest demand, he could optimize his workday by focusing on those areas. This would allow him to either earn more money in the same amount of time or save time for his family and personal life, ultimately improving his quality of life.*

- ***The purpose of this document is to provide a brief overview of the project.***

---

# 2_What Data Have I used?
- **2020-2022 Yellow Taxis data** : *I downloaded this dataset from [NYC Open Data](https://opendata.cityofnewyork.us/), a free public data source of __New York__ City.* 
*The dataset includes 17 fields such as `Pick-up time and date`,`Pick-up location`, `Drop-off location`, `Fare amount`, `Payment method` ..etc. you can have a look and review the full features description from the [data dictionary here](https://data.cityofnewyork.us/api/views/2upf-qytp/files/4a7a18af-bfc8-43d1-8a2e-faa503f75eb5?download=true&filename=data_dictionary_trip_records_yellow.pdf).*

- **Weather history dataset** : *I wanted to include some weather parameters such as `temp`, `precip` data to train the model, as it is sensible to think that rain would affect the taxi demand. I used __Web Scrapping__ and downloaded it from the [visual crossing website](https://www.visualcrossing.com/) for the period between Jan-2020 to Apr-2022. but it takes a couple of days for them to respond your query so you can also download it from my repo in the Data folder.*

- **Polygon shape file** : *In order to visualize the results I needed geometric data. This ``.shp`` file represents the boundaries zones for taxi pickups as delimited by the New York City __Taxi and Limousine Commission (TLC)__. You can download the file from several [websites](https://archive.nyu.edu/handle/2451/36743) or my Repo.*
  
![taxi_zone_map_manhattan](https://github.com/user-attachments/assets/6fee1df0-5d13-46f9-98ec-009d309a1518)

---

# 3_Methodology & Internal Structure
