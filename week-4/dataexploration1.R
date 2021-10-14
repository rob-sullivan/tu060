#we import our csv file dataset as a dataframe
df <- read.csv(file = 'dataset.csv')
#we look at some rows from our dataset
head(grades)
#we check to see their class type
sapply(grades, class)
#we loop through columns and convert columns that are characters into factors so we can use them as categories
for (i in colnames(grades)){
  if(class(grades[[i]])=="character"){
    grades[[i]] <- as.factor(grades[[i]])
  }
}
#we check that it worked
sapply(grades, class)
#we now examine a summary of our data
summary(grades)
plot(grades$school)


