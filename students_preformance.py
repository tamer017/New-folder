import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("StudentsPerformance.csv")
#remove the null columns and the columns that will not affect our calculations
# for knowing the culomns with null elements 
# print(data.isnull().sem())
# drop columns
#                              cleaning 
data=data.drop(["race/ethnicity","test preparation course"],axis=1)
#                              labeling
# extract the object columns and encode it
object_data=data.select_dtypes(include="object")
# encode it
encode=preprocessing.LabelEncoder()
for column in object_data.columns:
    object_data[column]=encode.fit_transform(object_data[column])
nonobject_data=data.select_dtypes(exclude="object")
data=pd.concat([object_data,nonobject_data],axis=1)
# find the relation between the columns 
c=data.corr()
# print(c)
sns.heatmap(c,annot=True)
plt.show()
# print(data)
