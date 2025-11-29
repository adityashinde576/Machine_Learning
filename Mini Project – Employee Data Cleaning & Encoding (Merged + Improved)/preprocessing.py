import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

data={
    'Name':['Amit','Sara','Ravi','Priya','John'],
    'Department':['IT','HR','Finance','IT','Finance'],
    'Experience':[2,5,None,3,4],
    'Salary':[40000,60000,50000,55000,None]
}

df=pd.DataFrame(data)
print("===Original Data===\n",df)

df['Experience'].fillna(df['Experience'].mean(),inplace=True)
df['Salary'].fillna(df['Salary'].mean(),inplace=True)
print("\n===Data After Filling Missing Values===\n",df)

le=LabelEncoder()
df['Dept_encoded']=le.fit_transform(df['Department'])
print("\nDepartment Mapping:",dict(zip(le.classes_,le.transform(le.classes_))))
print("\n===After Encoding Department===\n",df)

scaler=StandardScaler()
df[['Experience_scaled','Salary_scaled']]=scaler.fit_transform(df[['Experience','Salary']])
print("\n===Final Processed Data===\n",df)
