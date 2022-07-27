import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data = pd.read_csv("labeldata.csv")

#from sklearn.preprocessing import LabelEncoder
#labelencoder=LabelEncoder()
#Agri_data_Cols=data[['Area','Item_x']]
#for i in Agri_data_Cols:
#   data[i] = labelencoder.fit_transform(data[i])

y = data['yield']
X = data.drop(['yield','Domain Code', 'Domain_x',  'Element Code','Element_x',
         'Unit_x', 'Item Code','Area Code',
       'Domain_y', 'Element_y', 'Item_y',
       'Unit_y'], axis=1)



X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf_grid =RandomForestRegressor()
rf_grid.fit(X_train,y_train)
pickle.dump(rf_grid,open('model_yield.pkl','wb'))
