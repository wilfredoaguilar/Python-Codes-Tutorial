# What is Pandas?

" Pandas is a Python library used for working with data sets.
" It has functions for analyzing, cleaning, exploring, and manipulating data.""

" Learning by Examples

# Load a CSV file into a Pandas DataFrame:

import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string())



# Pandas as pd

" Pandas is usually imported under the pd alias.
" alias: In Python alias are an alternate name for referring to the same thing.
" Create an alias with the as keyword while importing:

" import pandas as pd
" Now the Pandas package can be referred to as pd instead of pandas.

import pandas

mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

myvar = pandas.DataFrame(mydataset)

print(myvar)



" Checking Pandas Version
" The version string is stored under __version__ attribute.

import pandas as pd

print(pd.__version__)



# What is a Series?
" A Pandas Series is like a column in a table.
" It is a one-dimensional array holding data of any type.
" Create a simple Pandas Series from a list:

import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a)

print(myvar)



# Labels
" If nothing else is specified, the values are labeled with their index number. 
" First value has index 0, second value has index 1 etc.
" This label can be used to access a specified value.
" Return the first value of the Series:

print(myvar[0])


# Create Labels
" With the index argument, you can name your own labels.
" Create you own labels:

import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)



# Key/Value Objects as Series
" You can also use a key/value object, like a dictionary, when creating a Series.
" Create a simple Pandas Series from a dictionary:

import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories)

print(myvar)


# To select only some of the items in the dictionary, 
" use the index argument and specify only the items you want to include in the Series.
" Create a Series using only data from "day1" and "day2":

import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])

print(myvar)


# DataFrames
" Data sets in Pandas are usually multi-dimensional tables, called DataFrames.
" Series is like a column, a DataFrame is the whole table.
" Create a DataFrame from two Series:

import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar = pd.DataFrame(data)

print(myvar)



# What is a DataFrame?
" A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array, 
" or a table with rows and columns.
" Create a simple Pandas DataFrame:

import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}


# load data into a DataFrame object:
df = pd.DataFrame(data)

print(df)



# Locate Row
" As you can see from the result above, the DataFrame is like a table with rows and columns.
" Pandas use the loc attribute to return one or more specified row(s)
" Return row 0:
" refer to the row index:
    
print(df.loc[0])


" Return row 0 and 1:
" use a list of indexes:
    
print(df.loc[[0, 1]])



# Named Indexes
" With the index argument, you can name your own indexes.
" Add a list of names to give each row a name:

import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df) 



# Locate Named Indexes
" Use the named index in the loc attribute to return the specified row(s).
" Return "day2":
" refer to the named index:
    
print(df.loc["day2"])



# Load Files Into a DataFrame
" If your data sets are stored in a file, Pandas can load them into a DataFrame.
" Load a comma separated file (CSV file) into a DataFrame:

import pandas as pd

df = pd.read_csv('data.csv')

print(df) 



# Load the CSV into a DataFrame:

import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 


# Tip: use to_string() to print the entire DataFrame.



# Read JSON
" Big data sets are often stored, or extracted as JSON.
" JSON is plain text, but has the format of an object, and is well known in the world of programming, including Pandas.


# Load the JSON file into a DataFrame:

import pandas as pd

df = pd.read_json('data.json')

print(df.to_string()) 


# Load a Python Dictionary into a DataFrame:

import pandas as pd

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df) 



# Viewing the Data
" One of the most used method for getting a quick overview of the DataFrame, is the head() method.
" Get a quick overview by printing the first 10 rows of the DataFrame:

import pandas as pd

df = pd.read_csv('data.csv')

print(df.head(10))



# Print the first 5 rows of the DataFrame:

import pandas as pd

df = pd.read_csv('data.csv')

print(df.head())



# The tail() method returns the headers and a specified number of rows, starting from the bottom.
" Print the last 5 rows of the DataFrame:

print(df.tail()) 



# Info About the Data
" The DataFrames object has a method called info(), that gives you more information about the data set.
" Print information about the data:

print(df.info()) 


# Data Cleaning
" Remove Rows
" One way to deal with empty cells is to remove rows that contain empty cells.
" This is usually OK, since data sets can be very big, 
" and removing a few rows will not have a big impact on the result.

# Return a new Data Frame with no empty cells:

import pandas as pd

df = pd.read_csv('data.csv')

new_df = df.dropna()

print(new_df.to_string())



"If you want to change the original DataFrame, use the inplace = True argument:

# Remove all rows with NULL values:

import pandas as pd

df = pd.read_csv('data.csv')

df.dropna(inplace = True)

print(df.to_string())


" Note: Now, the dropna(inplace = True) will NOT return a new DataFrame, 

" but it will remove all rows containg NULL values from the original DataFrame.



# Replace Empty Values
" Another way of dealing with empty cells is to insert a new value instead.
" This way you do not have to delete entire rows just because of some empty cells.
" The fillna() method allows us to replace empty cells with a value:

# Replace NULL values with the number 130:

import pandas as pd

df = pd.read_csv('data.csv')

df.fillna(130, inplace = True)



# Replace Only For Specified Columns
" The example above replaces all empty cells in the whole Data Frame.
" To only replace empty values for one column, specify the column name for the DataFrame:
" Replace NULL values in the "Calories" columns with the number 130:

import pandas as pd

df = pd.read_csv('data.csv')

df["Calories"].fillna(130, inplace = True)



# Replace Using Mean, Median, or Mode
" A common way to replace empty cells, is to calculate the mean, median or mode value of the column.
" Pandas uses the mean() median() and mode() methods to calculate the respective values for a specified column:

    
# Calculate the MEAN, and replace any empty values with it:

import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mean()

df["Calories"].fillna(x, inplace = True)

" Mean = the average value (the sum of all values divided by number of values).



# Calculate the MEDIAN, and replace any empty values with it:

import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].median()

df["Calories"].fillna(x, inplace = True)

" Median = the value in the middle, after you have sorted all values ascending.



# Calculate the MODE, and replace any empty values with it:

import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mode()[0]

df["Calories"].fillna(x, inplace = True)

" Mode = the value that appears most frequently.



# Data of Wrong Format
" Cells with data of wrong format can make it difficult, or even impossible, to analyze data.
" To fix it, you have two options: remove the rows, or convert all cells in the columns into the same format.


# Convert to date:

import pandas as pd

df = pd.read_csv('data.csv')

df['Date'] = pd.to_datetime(df['Date'])

print(df.to_string())


# Removing Rows
" The result from the converting in the example above gave us a NaT value, 
" which can be handled as a NULL value, and we can remove the row by using the dropna() method.


# Remove rows with a NULL value in the "Date" column:

df.dropna(subset=['Date'], inplace = True)


# Replacing Values
" One way to fix wrong values is to replace them with something else.

Set "Duration" = 45 in row 7:

df.loc[7, 'Duration'] = 45


" For small data sets you might be able to replace the wrong data one by one, but not for big data sets.
" To replace wrong data for larger data sets you can create some rules, e.g. set some boundaries for legal values, 
" and replace any values that are outside of the boundaries.


# Loop through all values in the "Duration" column.

" If the value is higher than 120, set it to 120:

for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120
    


# Removing Rows
" Another way of handling wrong data is to remove the rows that contains wrong data.
" This way you do not have to find out what to replace them with, 
" and there is a good chance you do not need them to do your analyses.

# Delete rows where "Duration" is higher than 120:

for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)
    

# Returns True for every row that is a duplicate, othwerwise False:

print(df.duplicated())


# Removing Duplicates
" To remove duplicates, use the drop_duplicates() method.

# Remove all duplicates:

df.drop_duplicates(inplace = True)



# Finding Relationships
" A great aspect of the Pandas module is the corr() method.
" The corr() method calculates the relationship between each column in your data set.

# Show the relationship between the columns:

df.corr()


# Perfect Correlation:
" We can see that "Duration" and "Duration" got the number 1.000000, which makes sense, 
" each column always has a perfect relationship with itself.

# Good Correlation:
" "Duration" and "Calories" got a 0.922721 correlation, which is a very good correlation, 
" and we can predict that the longer you work out, the more calories you burn, and the other way around: 
" if you burned a lot of calories, you probably had a long work out.

# Bad Correlation:
" Duration" and "Maxpulse" got a 0.009403 correlation, which is a very bad correlation, 
" meaning that we can not predict the max pulse by just looking at the duration of the work out, and vice versa.



# Plotting
" Pandas uses the plot() method to create diagrams.
" We can use Pyplot, a submodule of the Matplotlib library to visualize the diagram on the screen.

# Import pyplot from Matplotlib and visualize our DataFrame:

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot()

plt.show()


# Scatter Plot
" Specify that you want a scatter plot with the kind argument:

# kind = 'scatter'

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')

plt.show()


# A scatterplot where there are no relationship between the columns:

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot(kind = 'scatter', x = 'Duration', y = 'Maxpulse')

plt.show()



# Histogram
" Use the kind argument to specify that you want a histogram:

# kind = 'hist'

" A histogram needs only one column.


df["Duration"].plot(kind = 'hist')