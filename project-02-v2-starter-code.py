# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 15px; height: 80px">
#
# # Project 2
#
# ### Exploratory Data Analysis (EDA)
#
# ---
#
# Your hometown mayor just created a new data analysis team to give policy advice, and the administration recruited _you_ via LinkedIn to join it. Unfortunately, due to budget constraints, for now the "team" is just you...
#
# The mayor wants to start a new initiative to move the needle on one of two separate issues: high school education outcomes, or drug abuse in the community.
#
# Also unfortunately, that is the entirety of what you've been told. And the mayor just went on a lobbyist-funded fact-finding trip in the Bahamas. In the meantime, you got your hands on two national datasets: one on SAT scores by state, and one on drug use by age. Start exploring these to look for useful patterns and possible hypotheses!
#
# ---
#
# This project is focused on exploratory data analysis, aka "EDA". EDA is an essential part of the data science analysis pipeline. Failure to perform EDA before modeling is almost guaranteed to lead to bad models and faulty conclusions. What you do in this project are good practices for all projects going forward, especially those after this bootcamp!
#
# This lab includes a variety of plotting problems. Much of the plotting code will be left up to you to find either in the lecture notes, or if not there, online. There are massive amounts of code snippets either in documentation or sites like [Stack Overflow](https://stackoverflow.com/search?q=%5Bpython%5D+seaborn) that have almost certainly done what you are trying to do.
#
# **Get used to googling for code!** You will use it every single day as a data scientist, especially for visualization and plotting.
#
# #### Package imports

# +
import numpy as np
import scipy.stats as stats
import csv
import pandas as pd

# this line tells jupyter notebook to put the plots in the notebook rather than saving them to file.
# %matplotlib inline

# this line makes plots prettier on mac retina screens. If you don't have one it shouldn't do anything.
# %config InlineBackend.figure_format = 'retina'
# -

# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 1. Load the `sat_scores.csv` dataset and describe it
#
# ---
#
# You should replace the placeholder path to the `sat_scores.csv` dataset below with your specific path to the file.
#
# ### 1.1 Load the file with the `csv` module and put it in a Python dictionary
#
# The dictionary format for data will be the column names as key, and the data under each column as the values.
#
# Toy example:
# ```python
# data = {
#     'column1':[0,1,2,3],
#     'column2':['a','b','c','d']
#     }
# ```



# ### 1.2 Make a pandas DataFrame object with the SAT dictionary, and another with the pandas `.read_csv()` function
#
# Compare the DataFrames using the `.dtypes` attribute in the DataFrame objects. What is the difference between loading from file and inputting this dictionary (if any)?



# If you did not convert the string column values to float in your dictionary, the columns in the DataFrame are of type `object` (which are string values, essentially). 

# ### 1.3 Look at the first ten rows of the DataFrame: what does our data describe?
#
# From now on, use the DataFrame loaded from the file using the `.read_csv()` function.
#
# Use the `.head(num)` built-in DataFrame function, where `num` is the number of rows to print out.
#
# You are not given a "codebook" with this data, so you will have to make some (very minor) inference.



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 2. Create a "data dictionary" based on the data
#
# ---
#
# A data dictionary is an object that describes your data. This should contain the name of each variable (column), the type of the variable, your description of what the variable is, and the shape (rows and columns) of the entire dataset.



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 3. Plot the data using seaborn
#
# ---
#
# ### 3.1 Using seaborn's `distplot`, plot the distributions for each of `Rate`, `Math`, and `Verbal`
#
# Set the keyword argument `kde=False`. This way you can actually see the counts within bins. You can adjust the number of bins to your liking. 
#
# [Please read over the `distplot` documentation to learn about the arguments and fine-tune your chart if you want.](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.distplot.html#seaborn.distplot)



# ### 3.2 Using seaborn's `pairplot`, show the joint distributions for each of `Rate`, `Math`, and `Verbal`
#
# Explain what the visualization tells you about your data.
#
# [Please read over the `pairplot` documentation to fine-tune your chart.](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot)



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 4. Plot the data using built-in pandas functions.
#
# ---
#
# Pandas is very powerful and contains a variety of nice, built-in plotting functions for your data. Read the documentation here to understand the capabilities:
#
# http://pandas.pydata.org/pandas-docs/stable/visualization.html
#
# ### 4.1 Plot a stacked histogram with `Verbal` and `Math` using pandas



# ### 4.2 Plot `Verbal` and `Math` on the same chart using boxplots
#
# What are the benefits of using a boxplot as compared to a scatterplot or a histogram?
#
# What's wrong with plotting a box-plot of `Rate` on the same chart as `Math` and `Verbal`?



# <img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ### 4.3 Plot `Verbal`, `Math`, and `Rate` appropriately on the same boxplot chart
#
# Think about how you might change the variables so that they would make sense on the same chart. Explain your rationale for the choices on the chart. You should strive to make the chart as intuitive as possible. 
#



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 5. Create and examine subsets of the data
#
# ---
#
# For these questions you will practice **masking** in pandas. Masking uses conditional statements to select portions of your DataFrame (through boolean operations under the hood.)
#
# Remember the distinction between DataFrame indexing functions in pandas:
#
#     .iloc[row, col] : row and column are specified by index, which are integers
#     .loc[row, col]  : row and column are specified by string "labels" (boolean arrays are allowed; useful for rows)
#     .ix[row, col]   : row and column indexers can be a mix of labels and integer indices
#     
# For detailed reference and tutorial make sure to read over the pandas documentation:
#
# http://pandas.pydata.org/pandas-docs/stable/indexing.html
#
#
#
# ### 5.1 Find the list of states that have `Verbal` scores greater than the average of `Verbal` scores across states
#
# How many states are above the mean? What does this tell you about the distribution of `Verbal` scores?
#
#
#



# ### 5.2 Find the list of states that have `Verbal` scores greater than the median of `Verbal` scores across states
#
# How does this compare to the list of states greater than the mean of `Verbal` scores? Why?



# ### 5.3 Create a column that is the difference between the `Verbal` and `Math` scores
#
# Specifically, this should be `Verbal - Math`.



# ### 5.4 Create two new DataFrames showing states with the greatest difference between scores
#
# 1. Your first DataFrame should be the 10 states with the greatest gap between `Verbal` and `Math` scores where `Verbal` is greater than `Math`. It should be sorted appropriately to show the ranking of states.
# 2. Your second DataFrame will be the inverse: states with the greatest gap between `Verbal` and `Math` such that `Math` is greater than `Verbal`. Again, this should be sorted appropriately to show rank.
# 3. Print the header of both variables, only showing the top 3 states in each.



# ## 6. Examine summary statistics
#
# ---
#
# Checking the summary statistics for data is an essential step in the EDA process!
#
# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ### 6.1 Create the correlation matrix of your variables (excluding `State`).
#
# What does the correlation matrix tell you?
#



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ### 6.2 Use pandas'  `.describe()` built-in function on your DataFrame
#
# Write up what each of the rows returned by the function indicate.



# <img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ### 6.3 Assign and print the _covariance_ matrix for the dataset
#
# 1. Describe how the covariance matrix is different from the correlation matrix.
# 2. What is the process to convert the covariance into the correlation?
# 3. Why is the correlation matrix preferred to the covariance matrix for examining relationships in your data?



# <img src="http://imgur.com/l5NasQj.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 7. Performing EDA on "drug use by age" data.
#
# ---
#
# You will now switch datasets to one with many more variables. This section of the project is more open-ended - use the techniques you practiced above!
#
# We'll work with the "drug-use-by-age.csv" data, sourced from and described here: https://github.com/fivethirtyeight/data/tree/master/drug-use-by-age.
#
# ### 7.1
#
# Load the data using pandas. Does this data require cleaning? Are variables missing? How will this affect your approach to EDA on the data?



# ### 7.2 Do a high-level, initial overview of the data
#
# Get a feel for what this dataset is all about.
#
# Use whichever techniques you'd like, including those from the SAT dataset EDA. The final response to this question should be a written description of what you infer about the dataset.
#
# Some things to consider doing:
#
# - Look for relationships between variables and subsets of those variables' values
# - Derive new features from the ones available to help your analysis
# - Visualize everything!



# ### 7.3 Create a testable hypothesis about this data
#
# Requirements for the question:
#
# 1. Write a specific question you would like to answer with the data (that can be accomplished with EDA).
# 2. Write a description of the "deliverables": what will you report after testing/examining your hypothesis?
# 3. Use EDA techniques of your choice, numeric and/or visual, to look into your question.
# 4. Write up your report on what you have found regarding the hypothesis about the data you came up with.
#
#
# Your hypothesis could be on:
#
# - Difference of group means
# - Correlations between variables
# - Anything else you think is interesting, testable, and meaningful!
#
# **Important notes:**
#
# You should be only doing EDA _relevant to your question_ here. It is easy to go down rabbit holes trying to look at every facet of your data, and so we want you to get in the practice of specifying a hypothesis you are interested in first and scoping your work to specifically answer that question.
#
# Some of you may want to jump ahead to "modeling" data to answer your question. This is a topic addressed in the next project and **you should not do this for this project.** We specifically want you to not do modeling to emphasize the importance of performing EDA _before_ you jump to statistical analysis.

# ** Question and deliverables**
#
#
# ...

# +
# Code
# -

# **Report**
#
#
#
# ...

# <img src="http://imgur.com/xDpSobf.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ## 8. Introduction to dealing with outliers
#
# ---
#
# Outliers are an interesting problem in statistics, in that there is not an agreed upon best way to define them. Subjectivity in selecting and analyzing data is a problem that will recur throughout the course.
#
# 1. Pull out the rate variable from the sat dataset.
# 2. Are there outliers in the dataset? Define, in words, how you _numerically define outliers._
# 3. Print out the outliers in the dataset.
# 4. Remove the outliers from the dataset.
# 5. Compare the mean, median, and standard deviation of the "cleaned" data without outliers to the original. What is different about them and why?



# <img src="http://imgur.com/GCAf1UX.png" style="float: left; margin: 25px 15px 0px 0px; height: 25px">
#
# ### 9. Percentile scoring and spearman rank correlation
#
# ---
#
# ### 9.1 Calculate the spearman correlation of sat `Verbal` and `Math`
#
# 1. How does the spearman correlation compare to the pearson correlation? 
# 2. Describe clearly in words the process of calculating the spearman rank correlation.
#   - Hint: the word "rank" is in the name of the process for a reason!
#



# ### 9.2 Percentile scoring
#
# Look up percentile scoring of data. In other words, the conversion of numeric data to their equivalent percentile scores.
#
# http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html
#
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html
#
# 1. Convert `Rate` to percentiles in the sat scores as a new column.
# 2. Show the percentile of California in `Rate`.
# 3. How is percentile related to the spearman rank correlation?



# ### 9.3 Percentiles and outliers
#
# 1. Why might percentile scoring be useful for dealing with outliers?
# 2. Plot the distribution of a variable of your choice from the drug use dataset.
# 3. Plot the same variable but percentile scored.
# 4. Describe the effect, visually, of coverting raw scores to percentile.


