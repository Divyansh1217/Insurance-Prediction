import streamlit as st
import pandas as pd

data=pd.read_csv("TravelInsurancePrediction.csv")

data.drop(columns=["Unnamed: 0"],inplace=True ,axis=0)
data["TravelInsurance"]=data["TravelInsurance"].map({0:"Not Purchased",1:"Purchased"})

import plotly.express as px
figure = st.plotly_chart(px.histogram(data, x = "Age", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Age"))
figure = st.plotly_chart(px.histogram(data, x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Income"))

import numpy as np
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
accuracy=accuracy_score(ytest,predictions)
st.write("accuracy of model",accuracy*100,"%")
