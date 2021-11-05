# basic commands
#this is a comment
#save.image("testfile")
#q() will quit
#xdata <- c(1:6)
#ls
#xdata
#library("MASS")
#install.packages("ks")


#Basic Statistics
xdata <- c(1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 8)
#calculate median and mean
median(xdata)
mean(xdata)
#calculate mode
# r does not have mode command so we need to create a function.
getMode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
getMode(xdata)
# open cardata.R and run commands (i,k,b and p should be in memory)
cardata = data.frame(i, k, b, p)
names(cardata) <- c('ID', 'Colour', 'Make', 'NCTPassed')
head(cardata)
cardata[c('Colour','Make')]
cardata[c(3:4)]
cardata$Make
#order chart
cardata$Make <- ordered(cardata$Make, levels=c('dacia','mazda', 'toyota', 'vw', 'mercedes', 'BMW'))
#now create a bar chart
counts <- table(cardata$Make)
counts
barplot(counts, main="Make distribution", xlab="Make", ylab="Count")
#histogram
# tut from https://www.datacamp.com/community/tutorials/make-histogram-basic-r#gs.o8GxTqU
chol <- read.table(url("http://assets.datacamp.com/blog_assets/chol.txt"), header = TRUE)
hist(AirPassengers)
hist(chol$AGE)

#hist(AirPassengers, 
 #    main="Histogram for Air Passengers", 
#     xlab="Passengers", 
 #    border="blue", 
  #   col="green",
   #  xlim=c(100,700),
  #   las=1, 
  #   breaks=5)

#hist(AirPassengers, breaks=c(100, 300, 500, 700)) 
#hist(AirPassengers, breaks=c(100, seq(200,700, 150)))
hist(AirPassengers, 
     main="Histogram for Air Passengers", 
     xlab="Passengers", 
     border="blue", 
     col="green", 
     xlim=c(100,700), 
     las=1, 
     breaks=5, 
     prob = TRUE)
lines(density(AirPassengers))
#inter-quartile range and box plot
IQR(chol$WEIGHT) #difference between the 75th and 25th percentiles
boxplot(WEIGHT~SMOKE, data=chol, main="weight", xlab="smokes", ylab="weight", ylim=c(0,120))
boxplot(AGE~SMOKE, data=chol, main="age", xlab="smokes", ylab="age")
boxplot(HEIGHT~SMOKE, data=chol, main="height", xlab="smokes", ylab="height")
var(chol$WEIGHT)
sd(chol$WEIGHT)
var(chol$HEIGHT)
sd(chol$HEIGHT)
plot(chol$WEIGHT, chol$CHOL, main="Scatterplot Example",
     xlab="Weight ", ylab="CHOL ", pch=1)
barplot(height=c(4,17,12,3),width=c(1,1,1,2),names=c('0-19','20-39','40-59','60-99'))
hist(c(3,5,7,17,21,21,23,25,25,28,30,31,32,33,33,33,34,36,37,38,39,41,44,44,45,47,49,
       50,50,52,55,56,57,62,66,70,76,84,91),breaks=c(0,20,40,60,100))

# find the range
dataset <- c(2, 3, 4, 4, 5, 5, 5, 7, 9, 10, 10, 11, 12, 12, 15)
max(dataset) - min(dataset)
IQR(dataset)


#P1a_description_tallies_Q3
data1 <- c(0,3,1,2,4)
barplot(height=data1, names="hello")

#P1a_description_tallies_Q4
measure_type <- c(rep("shoe_size", 5))
shoe_size_table <- table(data1, measure_type)
barplot(shoe_size_table, legend=rownames(shoe_size_table), args.legend=c(x="center"))
pie(shoe_size_table, labels=rownames(shoe_size_table))
boxplot(data1)
