# To study the topics presented here, you should open this file in RStudio and execute the R commands, 
# reading the comments as you go along. When this script file is open in R, you can execute a command by
# first clicking on line the command is on and then clicking the 'Run' button. You will see the command
# being copied and run automatically in the console below. In the console you will also see the result 
# of the command having been executed.
#
# Alternatively you can open 'plain' R and copy and paste the commands to execute them.

# NOTE: Before you start, put the file "LAB 2.2.2 Relationships DEMO.csv" provided with this 
# script into your 'Downloads' directory.

# READING IN A FILE
# ==================
# see the directory you are in
getwd()

# change the directory you are in - NOTE: you should put in the path to your downloads directory
setwd("~/Downloads")

# reading in a csv file
read.csv("R3a_ireland_transport.csv")

# reading in a csv file and assigning to variable df
df <- read.csv("R3a_ireland_transport.csv")

# printing variable df to screen
df

# WORKING WITH DATA TYPES (DATA FRAME AND VECTOR)
# ====================================================
# checking the type of variable df (it is a data frame)
class(df)

# extracting a column from the data frame as a vector
df$Mode

# now instead of just extracting that column, we extract it and assign it to the variable v
m <- df$Mode

# we print variable m to the screen
m

# check the type of m (it is a 'factor' which means 'categorical variable')
class(m)

# when we check the type of a vector, R doesn't say it is a 'vector' but 
# tells us the type of the data that is in the vector
#
# the following checks the type of a vector with numbers 1, 2 and 3 in it
class(c(1, 2, 3))
# the following checks the type of a vector with character strings "one", "two", "three" in it
class(c("one", "two", "three"))

# we can extract a sub-data-frame from a data frame 
# let's extract a sub-data-frame with columns "Mode" and "X1986" from df - 
# to extract a list of columns, we pass in a vector with column names or numbers
df[c("Mode", "X1986")]
# now let's try the same with column numbers
df[c(3, 4)]

# let's extract and assing to variable df2, then check type of df2 (it's also a data frame)
df2 <- df[c("Mode", "X1986")]
class(df2)

# we can pick out rows, for example 2, 4 and 6
# (if there are two arguments in the square brackets, the list of rows is the first argument 
# and the list of columns is the second argument)
df[c(2, 4, 6), c("Mode", "X1986")]
# we would have got the same result with the following (if the column list is left out, 
# they are all included)
df2[c(2, 4, 6),]

# WORKING WITH TABLES
# ==========================
# we can create a contingency table using the function xtabs():
# -------------
# for each unique combination of a Group value and a Sex value (for example
# 'Population aged 15 years and over at work' and 'Female'), all rows with that
# combination are found by the function and the numbers in those rows of column
# 'X1986' added up; the sums are presented in a contingency table
xtabs(df$X1986 ~ df$Group + df$Sex)

# if nothing is specified to the left side of '~', xtabs counts the number of 
# rows for each value combination and displays the counts in the table (note
# that the number 10 in all cells of the table results simply from the fact that
# each combination of Group and Sex is combined with 10 different modes of transport)
xtabs(~ df$Group + df$Sex)

# the same as above can be achieved with function table()
table(df$Group, df$Sex)

# a subset can be extracted from a table in the same way as with a data frame
# this function call sums up the counts for each of the unique combinations
# of Group and Sex values
ctab <- xtabs(df$X1986 ~ df$Group + df$Sex)
# print the table to screen
ctab
# and a subset of the table:
ctab[c(1, 3),]

# BOXPLOTS
# =====================
# For the boxplot examples we will use the built-in data set 'iris'.
#
# Print the 'iris' data set to screen (it contains dimensions for flowers belonging
# to three different iris species):
iris

# To create a boxplot for the variable 'Sepal.Length' we just call the boxplot() function:
boxplot(iris$Sepal.Length)

# To create side-by-side boxplots for the same variable ('Sepal.Length'), with values grouped 
# by the attribute 'Species', we call the function boxplot() in the following way:
boxplot(iris$Sepal.Length ~ iris$Species)

# TESTS
# =============
# To find the correlation between two numeric variables, we can call the function cor():
cor(iris$Sepal.Length, iris$Sepal.Width)

# The correlation for the whole set is not very high but if we extract a subset containing
# data for a single species, the picture will be different. To extract the 'setosa' subset
# from the 'iris' dataset we can use square brackets, as we did with the data frame, but 
# this time specifying the attribute values we would like to extract by (as we are extracting
# by rows i.e. all the rows where Species equals "setosa", we specify this requirement as the
# first argument and leave the second argument, after the comma, unspecified, meaning that we 
# don't want a subset of columns but all of them):
iris[iris$Species=="setosa",]

# Now lets assign this subset to a variable:
iris.setosa <- iris[iris$Species=="setosa",]

# And have a look at the variable:
iris.setosa

# create a scatterplot to visually assess the relatedness
plot(iris.setosa$Sepal.Length, iris.setosa$Sepal.Width)

# The correlation between 'Sepal.Length' and 'Sepal.Width' for this set is (we expect a high value):
cor(iris.setosa$Sepal.Length, iris.setosa$Sepal.Width)

# The correlation method used by default with cor() is Pearson's correlation coefficient, but we 
# can use, for example, Kendall's Tau:
cor(iris.setosa$Sepal.Length, iris.setosa$Sepal.Width, method="kendall")

# For more information on the function cor(), run:
?cor

# The function cor() gives us a correlation value, but if we want to understand the significance
# of the value, we can run a test:
cor.test(iris.setosa$Sepal.Length, iris.setosa$Sepal.Width)

# The p-value we get here is "p-value = 6.71e-10". How do we interpret this?
# The p-value is the probability that samples of the same size and with the same parameters
# would yield the same value for the correlation coefficient if the populations were unrelated.
# Because this probability is very low (6.71 times 10 to the power of -10!), we can say that
# there is a high level of certainty that the variables are correlated. The p-value for the
# while iris set is 0.1519, which doesn't even bring us to the 5% level of certainty, so we 
# conclude that there is no correlation.

# The next test we will perform is the chi-square test. Let's create the table of counts (see above)
# by Group and Sex and assign it to a variable:
gsTab <- xtabs(df$X1986 ~ df$Group + df$Sex)

# A quick look at what is in the table:
gsTab

# To run the Chi-square test:
chisq.test(gsTab)

# The Chi-square values is: X-squared = 62245 and p-value < 2.2e-16 (extremely low), indicating 
# that the Chi-square hypothesis of unrelatedness does not hold and that the counts in the different 
# groups will depend heavily on whether we are looking at the Male of Female column. Truly, the second
# row shows a marked difference i.e. there are more than twice as many males reported to be in 
# work than females (in 1986), so the result is not surprising.

# Just as a matter of interest, let's have a look at the same tests for 2016:
gsTab2 <- xtabs(df$X2016 ~ df$Group + df$Sex)
gsTab2
chisq.test(gsTab2)

# Now X-squared = 3125.3 but the p-value is still very low: p-value < 2.2e-16. Even though at first sight
# the numbers in gsTab2 look similar by column and row, there is a statistically significant discrepancy 
# between the numbers expected at random and those reported.

# Finally, lets try the Chi-squared test on a table with identical columns:
testTab <- cbind(c(1111, 2222, 3333), c(1111, 2222, 3333))
testTab
chisq.test(testTab)

# Now X-squared = 0 and p-value = 1. This would indicate perfect unrelatedness between the variable in 
# the row and the one in the column, a result that would be highly suspicious with any data collected
# from a natural or social process.






#****************************************************************************#
# Pearson's correlation coefficient
#****************************************************************************#
# We create two data vectors.
xdata<-c(2, 4, 4, 3, 3, 2, 2, 2, 2, 4)
ydata<-c(1, 4, 4, 1, 3, 2, 2, 2, 2, 7)

# The R function cor() calculates Pearson's correlation coefficient.
cor(xdata, ydata)

# To get more information, including the statistical significance of the correlation
# coefficient value, we can use the test:
cor.test(xdata, ydata)
# Interpretation of the result:

# Pearson's product-moment correlation
# 
# data:  xdata and ydata
# t = 3.4508, df = 8, p-value = 0.008685
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
# 0.2802397 0.9435585
# sample estimates:
# cor 
# 0.7734021 

# - t is the statistic tested in a T-test - in this case comparing the value of r with 0
# - df degrees of freedom (10 for the number of instances - 2 for the number of variables)
# - p-value is the probability for the calculated T-statistic to be greater than what was obtained here 
#   in the case that the actual correlation were 0 (in our case it is about 1% or 2% two-tailed) - 
#   this is derived from the T-distribution, which is used instead of the normal distribution when the
#   sample size is below 30.
#
# The 95 percent confidence interval is given for the value of r

#****************************************************************************#
# Kendall's Tau
#****************************************************************************#
# Let's look at some data that can be ordered but is not numeric. Weeate two data vectors reflecting 
# the rating of competitors in a competition by two judges. The instances here are the competitors, the
# attributes (variables) are the ratings given by the judges, by which the data is ordered for each
# of the judges.
judgeARatings <- c("A", "C", "B", "F", "D", "E", "G")
judgeBRatings <- c("A", "B", "C", "F", "E", "D", "G")

# Now we can use the Kendall's Tau to gauge the correlation between the two sets of ratings.
# The functions cor() and cor.test() work only with numeric values, so we will have to convert
# the vector of character string values to a vector of numeric values. This cannot be done directly
# but we can convert the vector of strings to a factor and then the factor to a vector of numeric values:
jARNum <- as.numeric(factor(judgeARatings))
jBRNum <- as.numeric(factor(judgeBRatings))

# Let's use the conversion to create numeric vectors for input to the cor() function.
cor(jARNum, jBRNum, method="kendall")
cor.test(jARNum, jBRNum, method="kendall")

# Interpretation of the result
#
# Kendall's rank correlation coefficient (Kendall's Tau) 
# 
# data:  jARNum and jBRNum
# T = 19, p-value = 0.01071
# alternative hypothesis: true tau is not equal to 0
# sample estimates:
# tau 
# 0.8095238 

# - T is the number of concordant pairs (number of pairs - number of discordant) = (7*6/2 - 2) = 19
# - p-value is the probability for the Tau from a sample like ours to have the value like we obtained
#   in the case that the real Tau for the population is 0 (it is about 1%, or 2% two-tailed for this example).


#****************************************************************************#
# Chi-squared test
#****************************************************************************#
# We prepare a data frame of patient counts, where the rows represent different 
# skin treatment types and the columns the level of recovery.
skin <- skin <- data.frame(None=c(20, 32, 8, 52), Partial=c(9, 72, 8, 32), Full=c(16, 64, 30, 12), row.names=c("Injection", "Tablet", "Laser", "Herbal"))
skin

# Now we perform the in-built chi-square test in R to see if the null hypothesis, 
# that there is no relationship between the type of treatment and its effectiveness, holds.
chisq.test(skin)

# interpretation of the result:
# X-squared = 66.166, df = 6, p-value = 2.492e-12
# X-squared is the calculated chi-squared value
# df is the calculated degrees of freedom ((4-1)(3-1)=6)
# p-value is the probability of chi-squared values falling above the calculated chi-squared value... in this case it is very low i.e. below any usual significance level e.g. 1%, so we can reject the hypothesis and conclude that there is a relationship between treatment type and outcome

