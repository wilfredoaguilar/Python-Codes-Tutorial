# In Machine Learning (and in mathematics) there are often three values that interests us:

" Mean - The average value
" Median - The mid point value
" Mode - The most common value


# To find the average speed: mean()

import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

print(x)



# To find the middle value: median()

import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.median(speed)

print(x)



# To find the number that appears the most: mode()

from scipy import stats

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = stats.mode(speed)

print(x)


# What is Standard Deviation?
" Standard deviation is a number that describes how spread out the values are.
" A low standard deviation means that most of the numbers are close to the mean (average) value.
" A high standard deviation means that the values are spread out over a wider range.


# To find the standard deviation: std()

import numpy

speed = [86,87,88,86,87,85,86]

x = numpy.std(speed)

print(x)


# Variance
" Variance is another number that indicates how spread out the values are.
" In fact, if you take the square root of the variance, you get the standard deviation!
" Or the other way around, if you multiply the standard deviation by itself, you get the variance!


#To find the variance: var()

import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.var(speed)

print(x)


# Standard deviation is the square root of the variance:

#To find the standard deviation:

import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.std(speed)

print(x)


# Symbols
" Standard Deviation is often represented by the symbol Sigma: s
" Variance is often represented by the symbol Sigma Square: s2


# What are Percentiles?
" Percentiles are used in statistics to give you a number that describes the value 
" that a given percent of the values are lower than.

#To find the percentiles:

import numpy

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

x = numpy.percentile(ages, 75)

print(x)


" What is the 75. percentile? The answer is 43, meaning that 75% of the people are 43 or younger.



# Data Distribution
" Earlier in this tutorial we have worked with very small amounts of data in our examples, 
" just to understand the different concepts.
" In the real world, the data sets are much bigger, but it can be difficult to gather real world 
" data, at least at an early stage of a project.


#Create an array containing 250 random floats between 0 and 5:

import numpy

x = numpy.random.uniform(0.0, 5.0, 250)

print(x)


# Histogram
" To visualize the data set we can draw a histogram with the data we collected.

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 250)

plt.hist(x, 5)
plt.show()


# Big Data Distributions
" An array containing 250 values is not considered very big, but now you know how to create a random 
" set of values, and by changing the parameters, you can create the data set as big as you want.


#Create an array with 100000 random numbers, and display them using a histogram with 100 bars:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()



# Normal Data Distribution
" In probability theory this kind of data distribution is known as the normal data distribution, or the Gaussian data distribution, after the mathematician Carl Friedrich Gauss who came up with the formula of this data distribution.

#A typical normal data distribution:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()


# Scatter Plot
" A scatter plot is a diagram where each value in the data set is represented by a dot.


# Use the scatter() method to draw a scatter plot diagram:

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()


# Random Data Distributions
" In Machine Learning the data sets can contain thousands-, or even millions, of values.
" You might not have real world data when you are testing an algorithm, you might have to use 
" randomly generated values.
" As we have learned in the previous chapter, the NumPy module can help us with that!
" Let us create two arrays that are both filled with 1000 random numbers from a normal data distribution.

" The first array will have the mean set to 5.0 with a standard deviation of 1.0.
" The second array will have the mean set to 10.0 with a standard deviation of 2.0:


# A scatter plot with 1000 dots:

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()


# Regression
" The term regression is used when you try to find the relationship between variables.
" In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.


# Linear Regression
" Linear regression uses the relationship between the data-points to draw a straight line through all them.
" This line can be used to predict future values.

" In the example below, the x-axis represents age, and the y-axis represents speed. We have registered the age 
" and speed of 13 cars as they were passing a tollbooth. Let us see if the data we collected 
" could be used in a linear regression:

# Start by drawing a scatter plot:

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()



# Import scipy and draw the line of Linear Regression:

import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


# R for Relationship
" It is important to know how the relationship between the values of the x-axis 
" and the values of the y-axis is, if there are no relationship the linear regression 
" can not be used to predict anything.

# This relationship - the coefficient of correlation - is called r.
" The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.

" Python and the Scipy module will compute this value for you, all you have to do 
" is feed it w"ith the x and y values.


# How well does my data fit in a linear regression?

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)


" Note: The result -0.76 shows that there is a relationship, not perfect, 
" but it indicates that we could use linear regression in future predictions.



# Predict Future Values

" Now we can use the information we have gathered to predict future values.

" Let us try to predict the speed of a 10 years old car.
" To do so, we need the same myfunc() function from the example above:

def myfunc(x):
  return slope * x + intercept


Predict the speed of a 10 years old car:

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(10)

print(speed)


# Bad Fit?
" Let us create an example where linear regression would not be the best method to predict future values.
" These values for the x- and y-axis should result in a very bad fit for linear regression:

import matplotlib.pyplot as plt
from scipy import stats

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()



" And the r for relationship?

# You should get a very low r value.

import numpy
from scipy import stats

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

"The result: 0.013 indicates a very bad relationship, and tells us that this data set 
" is not suitable for linear regression.



# Polynomial Regression
" If your data points clearly will not fit a linear regression 
" (a straight line through all data points), it might be ideal for polynomial regression.

" Polynomial regression, like linear regression, uses the relationship between the 
" variables x and y to find the best way to draw a line through the data points.


# How Does it Work?
" Python has methods for finding a relationship between data-points and to draw a line 
" of polynomial regression. We will show you how to use these methods instead of going through 
" the mathematic formula.

" In the example below, we have registered 18 cars as they were passing a certain tollbooth.
" We have registered the car's speed, and the time of day (hour) the passing occurred.

" The x-axis represents the hours of the day and the y-axis represents the speed:

    

# Start by drawing a scatter plot:

import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

plt.scatter(x, y)
plt.show()



# Import numpy and matplotlib then draw the line of Polynomial Regression:

import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()



# R-Squared
" It is important to know how well the relationship between the values of the x- and y-axis is, 
" if there are no relationship the polynomial regression can not be used to predict anything.

" The relationship is measured with a value called the r-squared.
" The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.

" Python and the Sklearn module will compute this value for you, all you have to do is 
" feed it with the x and y arrays:

    

#How well does my data fit in a polynomial regression?

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))


# Predict Future Values

" Now we can use the information we have gathered to predict future values.

" Let us try to predict the speed of a car that passes the tollbooth at around 17 P.M:

" To do so, we need the same mymodel array from the example above:

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))



# Predict the speed of a car passing at 17 P.M:

import numpy
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)


# Bad Fit?
" Let us create an example where polynomial regression would not be the best method 
" to predict future values.


# These values for the x- and y-axis should result in a very bad fit for polynomial regression:

import numpy
import matplotlib.pyplot as plt

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(2, 95, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()



# And the r-squared value?

# You should get a very low r-squared value.

import numpy
from sklearn.metrics import r2_score

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

print(r2_score(y, mymodel(x)))


" The result: 0.00995 indicates a very bad relationship, and tells us that this data set is 
" not suitable for polynomial regression.



# Multiple Regression
" Multiple regression is like linear regression, but with more than one independent value, 
" meaning that we try to predict a value based on two or more variables.

" We can predict the CO2 emission of a car based on the size of the engine, 
" but with multiple regression we can throw in more variables, like the weight of the car, 
" to make the prediction more accurate.


#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:

import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)


# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:

predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)


# Coefficient
" The coefficient is a factor that describes the relationship with an unknown variable.

" Example: if x is a variable, then 2x is x two times. x is the unknown variable, 
" and the number 2 is the coefficient.
" In this case, we can ask for the coefficient value of weight against CO2, 
" and for volume against CO2. The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values.



# Print the coefficient values of the regression object:

import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)



# Result Explained
" The result array represents the coefficient values of weight and volume.

" Weight: 0.00755095
" Volume: 0.00780526

" These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
" And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.

" I think that is a fair guess, but let test it!
" We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, 
" the CO2 emission will be approximately 107g.


# What if we increase the weight with 1000kg?

" Copy the example from before, but change the weight from 2300 to 3300:

import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[3300, 1300]])

print(predictedCO2)



# We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, will release 
" approximately 115 grams of CO2 for every kilometer it drives.

" Which shows that the coefficient of 0.00755095 is correct:

" 107.2087328 + (1000 * 0.00755095) = 114.75968



# Scale Features
" When your data has different values, and even different measurement units, it can be 
" difficult to compare them. What is kilograms compared to meters? Or altitude compared to time?

" The answer to this problem is scaling. We can scale data into new values that are easier to compare.
" Take a look at the table below, it is the same data set that we used in the multiple regression chapter,
" but this time the volume column contains values in liters instead of cm3 (1.0 instead of 1000).


" It can be difficult to compare the volume 1.0 with the weight 790, but if we scale them both 
" into comparable values, we can easily see how much one value is compared to the other.

" There are different methods for scaling data, in this tutorial we will use a method called standardization.


" The standardization method uses this formula:

z = (x - u) / s

" Where z is the new value, x is the original value, u is the mean and s is the standard deviation.

" If you take the weight column from the data set above, the first value is 790, and the scaled value will be:

(790 - 1292.23) / 238.74 = -2.1


" If you take the volume column from the data set above, the first value is 1.0, and the scaled value will be:

(1.0 - 1.61) / 0.38 = -1.59


" Now you can compare -2.1 with -1.59 instead of comparing 790 with 1.0.
" You do not have to do this manually, the Python sklearn module has a method 
" called StandardScaler() which returns a Scaler object with methods for transforming data sets.



# Scale all values in the Weight and Volume columns:

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("cars2.csv")

X = df[['Weight', 'Volume']]

scaledX = scale.fit_transform(X)

print(scaledX)



# Predict CO2 Values
" The task in the Multiple Regression chapter was to predict the CO2 emission from a car 
" when you only knew its weight and volume.
" When the data set is scaled, you will have to use the scale when you predict values:

# Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms:

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("cars2.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)


# Evaluate Your Model
" In Machine Learning we create models to predict the outcome of certain events, 
" like in the previous chapter where we predicted the CO2 emission of a car when we 
" knew the weight and engine size.



# To measure if the model is good enough, we can use a method called Train/Test.
" What is Train/Test
" Train/Test is a method to measure the accuracy of your model.

" It is called Train/Test because you split the the data set into two sets: a training set and a testing set.
" 80% for training, and 20% for testing.
" You train the model using the training set.
" You test the model using the testing set.

" Train the model means create the model.
" Test the model means test the accuracy of the model.

" Start With a Data Set
" Start with a data set you want to test.


# Our data set illustrates 100 customers in a shop, and their shopping habits.

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.show()


# Split Into Train/Test
" The training set should be a random selection of 80% of the original data.
" The testing set should be the remaining 20%.

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]


# Display the Training Set
" Display the same scatter plot with the training set:

plt.scatter(train_x, train_y)
plt.show()


# Display the Testing Set
" To make sure the testing set is not completely different, we will take a look at the 
" testing set as well.

plt.scatter(test_x, test_y)
plt.show()



# Fit the Data Set
" What does the data set look like? In my opinion I think the best fit would be a 
" polynomial regression, so let us draw a line of polynomial regression.

# To draw a line through the data points, we use the plot() method of the matplotlib module:

# Draw a polynomial regression line through the data points:

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()


" The result can back my suggestion of the data set fitting a polynomial regression, 
" even though it would give us some weird results if we try to predict values outside 
" of the data set. 

" Example: the line indicates that a customer spending 6 minutes in the shop would 
" make a purchase worth 200. That is probably a sign of overfitting.

" But what about the R-squared score? The R-squared score is a good indicator of 
" how well my data set is fitting the model.


# R2
" Remember R2, also known as R-squared?
" It measures the relationship between the x axis and the y axis, and the value ranges 
" from 0 to 1, where 0 means no relationship, and 1 means totally related.

" The sklearn module has a method called r2_score() that will help us find this relationship.
" In this case we would like to measure the relationship between the minutes a customer stays 
" in the shop and how much money they spend.



# How well does my training data fit in a polynomial regression?

import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print(r2)


" Note: The result 0.799 shows that there is a OK relationship.



# Bring in the Testing Set
" Now we have made a model that is OK, at least when it comes to training data.
" Now we want to test the model with the testing data as well, to see if gives us the same result.


# Let us find the R2 score when using testing data:

import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)


" Note: The result 0.809 shows that the model fits the testing set as well, and we are confident
" that we can use the model to predict future values.


# Predict Values
" Now that we have established that our model is OK, we can start predicting new values.
" How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?

print(mymodel(5))







