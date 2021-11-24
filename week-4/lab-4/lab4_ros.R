#we import our csv file dataset as a dataframe
grades <- read.csv(file = 'dataset.csv')
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

#we loop through our data and graph it
plot(grades$school) 
plot(grades$sex)         

#age
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$age
h<-hist(x, breaks=10, col="red", xlab="Age of Students",
        main="Age Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)
  
plot(grades$address)
plot(grades$famsize)
plot(grades$Pstatus)

#mother's education
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Medu
h<-hist(x, breaks=10, col="red", xlab="Education Level",
        main="Mother's Education Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#father's education
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Fedu
h<-hist(x, breaks=10, col="red", xlab="Education Level",
        main="Fathers's Education Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#mother's job
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Mjob
h<-hist(x, breaks=10, col="red", xlab="Job",
        main="Mother's Job Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

plot(grades$Mjob)  
plot(grades$Fjob)  
plot(grades$reason) 
plot(grades$nursery) 
plot(grades$internet) 


#MATHS 
plot(grades$guardian.m) 

#Home to school travel time
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$traveltime.m
h<-hist(x, breaks=10, col="red", xlab="Time",
        main="Travel time to school Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Weekly study time
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$studytime.m
h<-hist(x, breaks=10, col="red", xlab="Weekly Time",
        main="Study Time Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#failures
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$failures.m
h<-hist(x, breaks=10, col="red", xlab="Past Exam Failures [#]",
        main="Exam Failures Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)


plot(grades$schoolsup.m) 
plot(grades$famsup.m) 
plot(grades$paid.m)
plot(grades$activities.m) 
plot(grades$higher.m)
plot(grades$romantic.m)


#famrel.m
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$famrel.m
h<-hist(x, breaks=10, col="red", xlab="Quality of Relationship",
        main="Family Relationship Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#freetime.m
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$freetime.m
h<-hist(x, breaks=10, col="red", xlab="Time",
        main="Free time after school Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#goout.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$goout.m 
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Going out with friends Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Dalc.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Dalc.m
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Workday Alcohol Consumption Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Walc.m  
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Walc.m
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Weekend Alcohol Consumption Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#health.m  
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$health.m 
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Current Health Status Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#absences.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$absences.m
h<-hist(x, breaks=10, col="red", xlab="# times absent",
        main="Absenteeism Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#first period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$mG1
h<-hist(x, breaks=15, col="red", xlab="Score",
        main="First Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#second period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$mG2
h<-hist(x, breaks=10, col="red", xlab="Score",
        main="Second Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#third period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$mG3
h<-hist(x, breaks=10, col="red", xlab="Score",
        main="Third Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#PORTUGUESE LANGUAGE
plot(grades$guardian.p) 

#Home to school travel time
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$traveltime.m
h<-hist(x, breaks=10, col="red", xlab="Time",
        main="Travel time to school Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Weekly study time
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$studytime.p
h<-hist(x, breaks=10, col="red", xlab="Weekly Time",
        main="Study Time Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#failures
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$failures.p
h<-hist(x, breaks=10, col="red", xlab="Past Exam Failures [#]",
        main="Exam Failures Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

plot(grades$schoolsup.p) 
plot(grades$famsup.p) 
plot(grades$paid.p)
plot(grades$activities.p) 
plot(grades$higher.p)
plot(grades$romantic.p)


#famrel.m
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$famrel.p
h<-hist(x, breaks=10, col="red", xlab="Quality of Relationship",
        main="Family Relationship Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#freetime.m
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$freetime.p
h<-hist(x, breaks=10, col="red", xlab="Time",
        main="Free time after school Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#goout.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$goout.p 
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Going out with friends Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Dalc.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Dalc.p
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Workday Alcohol Consumption Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#Walc.m  
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$Walc.p
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Weekend Alcohol Consumption Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#health.m  
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$health.p
h<-hist(x, breaks=10, col="red", xlab="Level",
        main="Current Health Status Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#absences.m 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$absences.p
h<-hist(x, breaks=10, col="red", xlab="# times absent",
        main="Absenteeism Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#first period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$pG1
h<-hist(x, breaks=15, col="red", xlab="Score",
        main="First Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#second period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$pG2
h<-hist(x, breaks=10, col="red", xlab="Score",
        main="Second Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)

#third period grade 
# ref# https://www.statmethods.net/graphs/density.html
x <- grades$pG3
h<-hist(x, breaks=10, col="red", xlab="Score",
        main="Third Period Grade Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)



plot(grades[[3]])
hist(grades[[3]], horizontal=TRUE)
boxplot(grades[[0]], horizontal=TRUE) 
  

