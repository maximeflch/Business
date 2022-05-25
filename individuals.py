

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title= "Heart disease App for individuals",
    page_icon = "https://icon-library.com/images/heart-disease-icon/heart-disease-icon-7.jpg",
    layout="wide")


st.title("Heart disease App for individuals")
st.markdown("The purpose of this application is to predict the **risk** for heart diseases for individuals")
st.write("*Please answer the following questions so we can calculate your risk category!*")
@st.cache()
def load_data():
    original_data = pd.read_csv("heart_2020_cleaned.csv")
    return(original_data.dropna())

path= "E:/Business_Analytics/FinalApp/"

@st.cache()
def load_data2():
    data = pd.read_csv("new_customers_DecTr.csv")
    return(data.dropna())

@st.cache(allow_output_mutation=True)
def load_model():
    filename = 'model_forest03.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return(loaded_model)

@st.cache(allow_output_mutation=True)
def load_model2():
    filename = 'model_LogReg.sav'
    loaded_model2 = pickle.load(open(filename, 'rb'))
    return(loaded_model2)


# Load Data and Model
original_data = load_data()
data = load_data2()
model = load_model()
model2 = load_model2()

#Original Data
if st.sidebar.checkbox("Show original database", False):
    st.dataframe(original_data)


filtered_data = data.loc[[0]]
filtered_data.loc[:,:] = 0
#filtered_data = filtered_data.drop("HeartDisease_Yes", axis=1)

###First step: ask for age###
AgeCategory = data.iloc[:,9:21].columns
age = st.selectbox('How old are you?', AgeCategory )

#filtered_data = data.loc[data[age]==1, :]

filtered_data[age] = 1

#Propose checklist of different categories#

#SEX#


if st.checkbox("Male", False):
    filtered_data["Sex_Male"] = 1
elif st.checkbox("Female", False):
    filtered_data["Sex_Male"] = 0

#BMI#

decision = st.selectbox('Do you know your BMI?', ["Yes","No"])
if decision == "Yes":
    BMI = st.number_input('Enter your BMI:')
elif decision == "No":
    weight = st.number_input('Weight in kg:')
    height = st.number_input('Height in cm:')
    height = height/100.00
    height2 = height * height
    BMI = weight / (height2)
    BMI = round(BMI, 2)
    st.write(f"Your BMI is: **{BMI}**")

filtered_data['BMI'] = BMI

#SleepTime#
sleep = st.slider('How many hours do you sleep at night?',
                      1, 16)    
filtered_data["SleepTime"] = sleep   

#PhysicalHealth#
physical = st.slider('For how many days during the last 30 was your physical health not good?',
                      1, 30) 
filtered_data["PhysicalHealth"] = physical

#MentalHealth#

mental = st.slider('For how many days during the last 30 was your mental health not good?',
                      1, 30) 
filtered_data["MentalHealth"] = mental

#Smoking#

smoking = st.selectbox('Have you smoked at least 100 cigarettes in your entire life?', ["Yes","No"])
if smoking == "Yes":
    filtered_data["Smoking_Yes"] = 1
elif smoking == "No":
    filtered_data["Smoking_Yes"] = 0
    
#Alcohol#
if filtered_data["Sex_Male"].all() == 1:
    alcohol = st.selectbox('Are you having 14 or more drinks a week?', ["Yes","No"])
    if alcohol == "Yes":
        filtered_data["AlcoholDrinking_Yes"] = 1
    elif alcohol == "No":
        filtered_data["AlcoholDrinking_Yes"] = 0
else:
    alcohol = st.selectbox('Are you having 7 or more drinks a week?', ["Yes","No"])
    if alcohol == "Yes":
        filtered_data["AlcoholDrinking_Yes"] = 1
    elif alcohol == "No":
        filtered_data["AlcoholDrinking_Yes"] = 0

#Stroke#

stroke = st.selectbox('Have you ever had a stroke?', ["Yes","No"])
if stroke == "Yes":
    filtered_data["Stroke_Yes"] = 1
elif stroke == "No":
    filtered_data["Stroke_Yes"] = 0
    
#DiffWalking#

DiffWalking = st.selectbox('Do you have serious difficulty walking or climbing stairs?', ["Yes","No"])
if DiffWalking == "Yes":
    filtered_data["DiffWalking_Yes"] = 1
elif DiffWalking == "No":
    filtered_data["DiffWalking_Yes"] = 0
    
#Race#

Race_ = data.iloc[:,21:26].columns
race = st.selectbox('Imputed race/ethnicity value:', Race_ )
filtered_data[race] = 1   

#Diabetes#

Diabetes_ = data.iloc[:,26:29].columns
diabetes = st.selectbox('Do you have Diabetes?', Diabetes_ )
filtered_data[diabetes] = 1

#PhysicalActivity#
 
PhysicalActivity = st.selectbox('Have you been doing any physical activity or exercise during the last 30 days(other than you job)?', ["Yes","No"])
if PhysicalActivity == "Yes":
    filtered_data["PhysicalActivity_Yes"] = 1
elif PhysicalActivity == "No":
    filtered_data["PhysicalActivity_Yes"] = 0
    
#GenHealth#

GenHealth_ = data.iloc[:,30:34].columns
genhealth = st.selectbox('How would you judge your general health condition?', GenHealth_ )
filtered_data[genhealth] = 1

#Asthma#

Asthma = st.selectbox('Are you asthmatic?', ["Yes","No"])
if Asthma == "Yes":
    filtered_data["Asthma_Yes"] = 1
elif Asthma == "No":
    filtered_data["Asthma_Yes"] = 0

#KidneyDisease#

KidneyDisease = st.selectbox('Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?', ["Yes","No"])
if KidneyDisease == "Yes":
    filtered_data["KidneyDisease_Yes"] = 1
elif KidneyDisease == "No":
    filtered_data["KidneyDisease_Yes"] = 0

#SkinCancer#

SkinCancer = st.selectbox('Have you ever had a Skin Cancer?', ["Yes","No"])
if SkinCancer == "Yes":
    filtered_data["SkinCancer_Yes"] = 1
elif SkinCancer == "No":
    filtered_data["SkinCancer_Yes"] = 0

###Predict Risk###
if st.checkbox("Show client data", False):
    st.write(filtered_data)

estimation = filtered_data
estimation = model.predict_proba(estimation)


Risk = estimation[:,1].mean()
Risk = round(Risk, 2)
Risk = Risk*100.00

st.write(f'Your risk level to get a Heart Disease has been evaluated at {Risk}% !')
st.write("It's important to mention that this is just a prediction to determine the risk class. In fact the precision of the model is only at 0.39!")




