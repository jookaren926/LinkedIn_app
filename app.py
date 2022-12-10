#### LOCAL APP
#### LinkedIn User Prediction Application

#### Import packages

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





s = pd.read_csv("social_media_usage.csv") ## Load Data, Create Dataset


def clean_sm(x):
    return np.where(x == 1, 1, 0)


#### Feature engineering/selection

# Clean data
ss = pd.DataFrame({
    "sm_li" : np.where(s["web1h"] > 7, np.nan, 
                       np.where(clean_sm(s["web1h"]) == 1 , 1 ,0)),
    "income" : np.where(s["income"] <= 9, s["income"], np.nan),
    "education" : np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent" : np.where(s["par"] == 1, 1,
                        np.where(s["par"] == 2, 0 , np.nan)),
    "married":np.where(s["marital"] == 1, 1,
                       np.where(s["marital"] <7, 0, np.nan)),
    "female":np.where(s["gender"] == 2, 1,
                      np.where(s["gender"] == 1, 0, np.nan)),
    "age":np.where(s["age"] <= 97, s["age"], np.nan)})



# Drop missing data (or impute it)
ss = ss.dropna()

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility



#### Train model

# Initialize algorithm 
lr = LogisticRegression()

# Fit algorithm to training data
lr.fit(X_train, y_train)

#### income
st.markdown("### What is your income?")

income = st.selectbox("Income Range",
                options =  ["<$10K",
                            "$10k-$20k",
                            "$20k-$30k",
                            "$30k-$40k",
                            "$40k-$50k",
                            "$50k-$75k",
                            "$75k-$100k",
                            "$100k-$150k",
                            "$150k<"])


if income == "<$10K":
    income = 1
elif income == "$10k-$20k":
    income = 2
elif income == "$20k-$30k":
    income = 3
elif income == "$30k-$40k":
    income = 4
elif income == "$40k-$50k":
    income = 5
elif income == "$50k-$75k":
    income = 6
elif income == "$75k-$100k":
    income = 7
elif income == "$100k-$150k":
    income = 8
else:
    income = 9


#### educ
st.markdown("#### What is your highest education?")

educ = st.selectbox("Education Level",
                options =  ["Less than high school",
                            "High school incomplete",
                            "High school graduate",
                            "Some college, no degree",
                            "Two-year associate degree from a college or university",
                            "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                            "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                            "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"]
                            )


if educ == "Less than high school":
    educ = 1
elif educ == "High school incomplete":
    educ = 2
elif educ == "High school graduate":
    educ = 3
elif educ == "Some college, no degree":
    educ = 4
elif educ == "Two-year associate degree from a college or university":
    educ = 5
elif educ == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    educ = 6
elif educ == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ = 7
else :
    educ = 8

#### marital
st.markdown("### What is your marital status?")

marital = st.selectbox("Current marital Status",
                options =  ["Married",
                            "Not Married"]
                            )


if marital == "Married":
    marital = 1
else :
    marital = 0


#### Par
st.markdown("### Are you a parent?")

par = st.selectbox("Parent?",
                options =  ["Yes",
                            "No"]
                            )



if par == "Yes":
    par = 1
else :
    par = 0

#### Gender
st.markdown("### What is your gender?")

female = st.selectbox("Gender",
                options =  ["Female",
                            "Male"]
                            )


if female == "Female":
    female = 1
else :
    female = 0

## age
st.markdown("### How old are you?")
age = st.number_input('Age', step =1, max_value = 97)



#### Making predictions: in dataframe

# New data for features: "income", "education", "parent", "married", "female", "age"
person= [income, educ, par , marital , female, age]

# Predict class, given input features
predicted_class = lr.predict([person])

print(predicted_class)

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

print(probs)

if (st.button("submit")):
# Print predicted class and probability
    if predicted_class[0] == 1:
        st.text("You are a LinkedIn User")
    else : 
        st.text("You are NOT a LinkedIn User")
    st.text(f"Probability that this person is LinkedIn User: {probs[0][1]}")


