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
