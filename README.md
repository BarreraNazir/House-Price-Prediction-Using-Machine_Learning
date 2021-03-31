# House-Price-Prediction-Using-Machine-Learning

**Problem Identification**

  What are the things that a potential home buyer considers before purchasing a house? The location, the size of the property, vicinity to offices, schools, parks, restaurants, hospitals or the stereotypical white picket fence? What about the most important factor — the price? There is a  lack of trust in property developers in the city, housing units sold across India in 2017 dropped by 7 percent. For example, for a potential homeowner, over 9,000 apartment projects and flats for sale are available in the range of ₹42-52 lakh, followed by over 7,100 apartments that are in the ₹52-62 lakh budget segment, says a report by property website Makaan. Buying a home is a tricky choice, it is difficult to ascertain the price of a house in Bengaluru.

**Goal**

  The main goal of this project is to find the price of the house using their features.

**Get Started**

Importing Libraries and Data
Feature Scaling
Building Model

**Importing Libraries and Data**

  * Importing Libraries
  
      The task is performed using Pandas, NumpPy, MatplotLib, Seaborn and Sklearn Libraries.
  
  * Data
  
      The Boston dataset has been taken from kaggle where each row comprises one data-point and contains details about a plot. Various features affect the pricing of a house. The Boston housing dataset has 108 features out of which we'll use 107 to train the model. The 108th feature is the price, which we'll use as our target variable.

**Feature Scaling**

   Feature Scaling is done to normalize the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm.

**Building Model**

  We train the data on different machine learning algorithms, such as Linear Regression, XGBoost, and Support Vector Machine. XGBoost classifier gave better results than other algorithms.
