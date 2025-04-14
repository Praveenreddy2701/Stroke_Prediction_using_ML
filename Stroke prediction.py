import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df = pd.read_csv("C:/Desktop/healthcare-dataset-stroke-data.csv")

## initial data exploration##
print(df.info())
print(df.describe())
df.head()
df.tail()
df.dtypes

#to check null values##
df.isnull().sum()

#Drop id column##
df.drop('id', axis=1, inplace=True) 
df
##handling missing values with mean ##
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

#after replace to check missing values##
df.isnull().sum()
# to check duplicates##
df.duplicated().sum()

df.describe()

#Detecting outliers and inliers

numeric_cols=['age','hypertension','heart_disease','avg_glucose_level','bmi']
outliers=[]
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 *IQR
    outliers.extend(df[(df[col]<lower_bound)|(df[col]>upper_bound)].index)
outliers
S= df.drop(set(outliers))   
S
S.describe()

###Copy the cleaned DataFrame for transformation
df1 = S.copy()

from sklearn.preprocessing import StandardScaler

# Standardize numerical columns
scaler = StandardScaler()
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols])


print(df1[numeric_cols].head())

# Set style
sns.set(style="whitegrid")

# Plot before and after transformation
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Original distributions
original_data = df[['age', 'avg_glucose_level', 'bmi']]
for i, col in enumerate(original_data.columns):
    sns.histplot(original_data[col], kde=True, ax=axes[0, i], color="skyblue")
    axes[0, i].set_title(f"Original: {col}")

# Transformed distributions
df2 = df1[['age', 'avg_glucose_level', 'bmi']]
for i, col in enumerate(df2.columns):
    sns.histplot(df2[col], kde=True, ax=axes[1, i], color="salmon")
    axes[1, i].set_title(f"Transformed: {col}")

plt.tight_layout()
plt.show()

## distribution for each numerical column
num_cols = ['age', 'avg_glucose_level', 'bmi']
for col in num_cols:
    sns.histplot(S[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
#box plot for bmi
for col in numeric_cols:
    sns.boxplot(y=col, data=S)
    plt.title(f'Boxplot of {col}')
    plt.show()
#count of smoking in each catogery
df1['smoking_status'].value_counts() 

  ## gender by category## 
df1['gender'].value_counts()

  ##bar graph stroke###  
df1['stroke'].value_counts(normalize=True).plot(kind='bar')
plt.title("Class Balance: Stroke")
plt.xlabel("Stroke")
plt.ylabel("Proportion")
plt.show()
    


cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    sns.countplot(x=col, data=S)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()
    
    
    
#####ANALYSIS###

##1.###stroke cases differ between males and females###
S.groupby('gender')['stroke'].mean()
sns.countplot(data=S, x='gender', hue='stroke')
plt.show()

#2## Visualize the age distribution for stroke and non-stroke cases
sns.histplot(data=df1, x='age', hue='stroke', bins=30, kde=True)

##3### Create a boxplot to compare glucose levels for stroke vs non-stroke cases
sns.boxplot(data=S, x='stroke', y='avg_glucose_level')

###4!## Calculate stroke rates by smoking status
S.groupby('smoking_status')['stroke'].mean()

# Visualize stroke counts across smoking categories
sns.countplot(data=df1, x='smoking_status', hue='stroke')


import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("C:/Desktop/healthcare-dataset-stroke-data.csv")
# Find all indices where stroke is 0
zero_indices = df[df['stroke'] == 0].index

# Randomly select 800 indices from the zero_indices to change to 1
np.random.seed(42)  # for reproducibility
indices_to_change = np.random.choice(zero_indices, size=800, replace=False)

# Update the stroke column
df.loc[indices_to_change, 'stroke'] = 1

# Check the new distribution
stroke_counts_after = df['stroke'].value_counts()
stroke_counts_after
##5### Calculate stroke rate by hypertension status
df.groupby('hypertension')['stroke'].mean()

##9### Define a function to categorize BMI
def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

# Apply the function to create a new 'bmi_cat' column
df['bmi_cat'] = df['bmi'].apply(bmi_category)

# Calculate stroke rates by BMI category
df.groupby('bmi_cat')['stroke'].mean().plot(kind='bar')


# Stroke distribution within each gender using pie chart
import matplotlib.pyplot as plt

genders = df['gender'].unique()
for gender in genders:
    subset = df[df['gender'] == gender]
    labels = ['No Stroke', 'Stroke']
    sizes = subset['stroke'].value_counts().sort_index()
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title(f'Stroke Distribution in {gender}')
    plt.show()
    


# Line plot: Stroke rate vs age
df_sorted = df.sort_values(by='age')
age_groups = df_sorted.groupby('age')['stroke'].mean()

plt.figure(figsize=(10, 5))
plt.plot(age_groups.index, age_groups.values, color='teal')
plt.title('Stroke Rate by Age')
plt.xlabel('Age')
plt.ylabel('Stroke Rate')
plt.grid(True)
plt.show()
    
# Stroke percentage by work type
work_group = df.groupby('work_type')['stroke'].mean().sort_values()

plt.figure(figsize=(8, 5))
work_group.plot(kind='barh', color='orchid')
plt.xlabel('Stroke Rate')
plt.title('Stroke Rate by Work Type')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.show()    

###linear regression###
X = df1[['age']]
y = df1['bmi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict
check = pd.DataFrame({'age': [0.5]})  # Age is normalized (0 to 1)
result = model.predict(check)
print("Predicted bmi for Normalized Age 0.5:", result)
  
# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.xlabel('Age (Normalized)')
plt.ylabel('Avg Glucose Level (Normalized)')
plt.title('Linear Regression Line')
plt.grid(True)
plt.show()  

###KNN model ##
le = LabelEncoder()
categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical:df1[col] = le.fit_transform(df1[col])

scaler = MinMaxScaler()
features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
S_n = pd.DataFrame(scaler.fit_transform(S[features]), columns=features)
S_n.describe()

# Split data
x_train = S_n.iloc[:2944]
x_test = S_n.iloc[2944:]
y_train = S['ever_married'].iloc[:2944]
y_test = S['ever_married'].iloc[2944:]

# Apply KNN
knn = KNeighborsClassifier(n_neighbors=32)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

##Convert to int if necessary
# Replace string labels with integers
y_test = y_test.replace({'Yes': 1, 'No': 0})
y_pred = pd.Series(y_pred).replace({'Yes': 1, 'No': 0})

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.title("Confusion Matrix - Stroke Prediction (KNN)")
plt.grid(False)
plt.show()
