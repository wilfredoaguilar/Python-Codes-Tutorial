# Python sample for Data Science Tutorial

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data = pd.read_csv("data.csv", header=0, sep=",")

x = full_health_data["Average_Pulse"]
y = full_health_data["Calorie_Burnage"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Average_Pulse")
plt.ylabel ("Calorie_Burnage")
plt.show()


#  Samples


# import panda
import pandas as pd


# reading csv file
# header=0 means that the headers for the variable names are to be found in the first row (note that 0 means the first row in Python)
# sep="," means that "," is used as the separator between the values. This is because we are using the file type .csv (comma separated values)

health_data = pd.read_csv("data.csv", header=0, sep=",")


#print
print(health_data)


#Remove Blank Rows

health_data.dropna(axis=0,inplace=True)

print(health_data)


# Showing Data Types

print(health_data.info())



# To convert the data into float64

health_data["Average_Pulse"] = health_data['Average_Pulse'].astype(float)
health_data["Max_Pulse"] = health_data["Max_Pulse"].astype(float)

print (health_data.info())


# Analyze the Data
" When we have cleaned the data set, we can start analyzing the data.

# We can use the describe() function in Python to summarize data:


print(health_data.describe())



# Plot the Existing Data in Python
" First plot the values of Average_Pulse against Calorie_Burnage using the matplotlib library.

import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='line'),
plt.ylim(ymin=0)
plt.xlim(xmin=0)

plt.show()



# Find The Slope
" The slope is defined as how much calorie burnage increases

# Use Python to Find the Slope

def slope(x1, y1, x2, y2):
  s = (y2-y1)/(x2-x1)
  return s

print (slope(80,240,90,260))



# Find The Intercept
" The intercept is used to fine tune the functions ability to predict Calorie_Burnage


# Find the Slope and Intercept Using Python
" The np.polyfit() function returns the slope and intercept


import numpy as np

health_data = pd.read_csv("data.csv", header=0, sep=",")

x = health_data["Average_Pulse"]
y = health_data["Calorie_Burnage"]
slope_intercept = np.polyfit(x,y,1)

print(slope_intercept)



# Plot a New Graph in Python

import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='line'),
plt.ylim(ymin=0, ymax=400)
plt.xlim(xmin=0, xmax=150)

plt.show()



# Find the 10% percentile for Max_Pulse
" Percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than.


import numpy as np

Max_Pulse= full_health_data["Max_Pulse"]
percentile10 = np.percentile(Max_Pulse, 10)
print(percentile10)


" The 10% percentile of Max_Pulse is 120. This means that 10% of all the training sessions have a Max_Pulse of 120 or lower.



# Standard Deviation
" Standard deviation is a number that describes how spread out the observations are.

import numpy as np

std = np.std(full_health_data)
print(std)



# Coefficient of Variation
" The coefficient of variation is used to get an idea of how large the standard deviation is.

import numpy as np

cv = np.std(full_health_data) / np.mean(full_health_data)
print(cv)



# Variance
" Use Python to Find the Variance of health_data
" Variance is another number that indicates how spread out the values are.

import numpy as np

var = np.var(health_data)
print(var)


# Use Python to Find the Variance of Full Data Set

import numpy as np

var_full = np.var(full_health_data)
print(var_full)


# Correlation Coefficient

import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='scatter')
plt.show()


# The correlation coefficient measures the relationship between two variables.
" Example of a Perfect Linear Relationship (Correlation Coefficient = 1)
" We will use scatterplot to visualize the relationship between Average_Pulse 
" and Calorie_Burnage (we have used the small data set of the sports watch with 10 observations).



# This time we want scatter plots, so we change kind to "scatter":

import matplotlib.pyplot as plt

health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='scatter')
plt.show()


# Example of a Perfect Negative Linear Relationship (Correlation Coefficient = -1)
" We have plotted fictional data here. The x-axis represents the amount of hours worked 
" at our job before a training session. The y-axis is Calorie_Burnage.

" If we work longer hours, we tend to have lower calorie burnage because we are exhausted 
" before the training session.


# The correlation coefficient here is -1.

import pandas as pd
import matplotlib.pyplot as plt

negative_corr = {'Hours_Work_Before_Training': [10,9,8,7,6,5,4,3,2,1],
'Calorie_Burnage': [220,240,260,280,300,320,340,360,380,400]}
negative_corr = pd.DataFrame(data=negative_corr)

negative_corr.plot(x ='Hours_Work_Before_Training', y='Calorie_Burnage', kind='scatter')
plt.show()



# Example of No Linear Relationship (Correlation coefficient = 0)
" Here, we have plotted Max_Pulse against Duration from the full_health_data set.
" As you can see, there is no linear relationship between the two variables. 
" It means that longer training session does not lead to higher Max_Pulse.


# The correlation coefficient here is 0.


import matplotlib.pyplot as plt

full_health_data.plot(x ='Duration', y='Max_Pulse', kind='scatter')
plt.show()




# Correlation Matrix
" A matrix is an array of numbers arranged in rows and columns.
" A correlation matrix is simply a table showing the correlation coefficients between variables.


# Correlation Matrix in Python
" We can use the corr() function in Python to create a correlation matrix. 
" We also use the round() function to round the output to two decimals:


Corr_Matrix = round(full_health_data.corr(),2)
print(Corr_Matrix)


# Using a Heatmap
" We can use a Heatmap to Visualize the Correlation Between Variables:

import matplotlib.pyplot as plt
import seaborn as sns

correlation_full_health = full_health_data.corr()

axis_corr = sns.heatmap(
correlation_full_health,
vmin=-1, vmax=1, center=0,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()


# Data Science - Statistics Correlation vs. Causality
" Correlation Does Not Imply Causality
" Correlation measures the numerical relationship between two variables.
" A high correlation coefficient (close to 1), does not mean that we can for sure 
" conclude an actual relationship between two variables.
" There is an important difference between correlation and causality:

" Correlation is a number that measures how closely the data are related
" Causality is the conclusion that x causes y.


import pandas as pd
import matplotlib.pyplot as plt

Drowning_Accident = [20,40,60,80,100,120,140,160,180,200]
Ice_Cream_Sale = [20,40,60,80,100,120,140,160,180,200]
Drowning = {"Drowning_Accident": [20,40,60,80,100,120,140,160,180,200],
"Ice_Cream_Sale": [20,40,60,80,100,120,140,160,180,200]}
Drowning = pd.DataFrame(data=Drowning)

Drowning.plot(x="Ice_Cream_Sale", y="Drowning_Accident", kind="scatter")
plt.show()

correlation_beach = Drowning.corr()
print(correlation_beach)



