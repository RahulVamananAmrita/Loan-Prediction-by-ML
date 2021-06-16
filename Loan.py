import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Loan.csv')

x=df[['age','credit-rating','children']]
y=df.loan

regressor = LinearRegression()
regressor.fit(x,y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

y_predict = regressor.predict(X_test)
y_trainPredict=regressor.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

print("R2 score : %.2f" % r2_score(y_test,y_predict))

print("Test Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test,y_predict)))

print("Train Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_train,y_trainPredict)))

print("Test MAE: ", mean_absolute_error(y_test,y_predict))

print("Train MAE: ", mean_absolute_error(y_train, y_trainPredict))

import streamlit as st
from PIL import Image

st.title('Loan Prediction APP')
st.sidebar.header('Enter Your Details to calculate your Loan Eligibility Amount')

def user_report():
     Age = st.sidebar.slider('Age', 35,60, 24 )
     creditScore = st.sidebar.slider('credit rating', 0,45, 20 )
     children = st.sidebar.slider('children', 0,1, 5 )

     user_report_data = {
      'Age':Age,
      'credit-rating':creditScore,
      'children':children
      }

     report_data = pd.DataFrame(user_report_data, index=[0])
     return report_data

user_data = user_report()

st.header('Applicants Data')
st.write(user_data)

Loan_amount = regressor.predict(user_data)
st.subheader('Your Eligible Loan Amount ')
st.subheader('Rs '+str(np.round(Loan_amount[0], 2)))
       




