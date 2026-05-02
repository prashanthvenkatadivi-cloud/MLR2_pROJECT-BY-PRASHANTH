'''
In this file we are going to load the data and develop MLR code in oops concept
'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle

class MLR:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            print(self.df)
            self.df['city'] = self.df['city'].map(
                {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4, 'Mapple Valley': 5, 'Auburn': 6,
                 'North Bend': 7, 'Lake Forest Park': 8, 'Sammamish': 9, 'Des Moines': 10, 'Bothell': 11})
            self.X = self.df.iloc[:, : --12]  # independent
            self.y = self.df.iloc[:, -12]  # dependent
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.2,
                                                                                    random_state=48)
            print(f"Training dataset size: {len(self.X_train)} : {len(self.y_train)}")
            print(f"Testing dataset size: {len(self.X_test)} : {len(self.y_test)}")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")

    def training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train, self.y_train)
            self.y_train_predections = self.reg.predict(self.X_train)
            print(f"Train Accuracy: {r2_score(self.y_train, self.y_train_predections)}")
            print(f"Test Accuracy: {root_mean_squared_error(self.y_train, self.y_train_predections)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")


    def testing(self):
        try:
            self.y_test_predictions = self.reg.predict(self.X_test)
            print(f"Test Accuracy: {r2_score(self.y_test, self.y_test_predictions)}")
            print(f"Test Loss: {root_mean_squared_error(self.y_test, self.y_test_predictions)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")

    def check_own_data(self):
        try:
            bedrooms = 2
            bathrooms = 2
            sqft_lot = 1050
            sqft_living = 1.5
            floors = 2
            sqft_above = 1000
            sqft_basement = 3004
            yr_built = 1900
            yr_renovated = 1920
            condition = 3
            view = 2
            waterfront = 1
            print(f"Test Point Predictions : {self.reg.predict([[bedrooms, bathrooms, sqft_lot, sqft_living, floors, sqft_above, sqft_basement, yr_built, yr_renovated, condition, view, waterfront]])[0]}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")

    def saving_model(self):
        try:
            with open ("Model.pkl","wb") as f:
                pickle.dump(self.reg, f)

            print(f"--------Load and check--------")
            with open("Model.pkl","rb") as t:
                model = pickle.load(t)
                bedrooms = 2
                bathrooms = 2
                sqft_lot = 1050
                sqft_living = 1.5
                floors = 2
                sqft_above = 1000
                sqft_basement = 3004
                yr_built = 1900
                yr_renovated = 1920
                condition = 3
                view = 2
                waterfront = 1
                print(f"Loaded Model Predictions : {model.predict([[bedrooms, bathrooms, sqft_lot, sqft_living, floors, sqft_above, sqft_basement, yr_built, yr_renovated, condition, view, waterfront]])[0]}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")



if __name__ == '__main__':
    try:
        path = 'data.csv'
        obj = MLR(path)
        obj.training()
        obj.testing()
        obj.check_own_data()
        obj.saving_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg} ")
