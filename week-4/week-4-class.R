#download and unzip file into a bankdata folder
url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
download.file(url, destfile = 'bank.zip')
unzip('bank.zip', exdir='./bankData')

#use readr library and read in csv file
library(readr)
#bankData <- read_delim('./bankData/bank-full.csv', delim=';', na=c('unknown'), col_types = 'nccccncccncnnnncc')
bankData <- read_delim('./bankData/bank-full.csv', delim=';', na=c('unknown'), guess_max=50000)


#basic checks
problems(bankData)
nrow(bankData)
ncol(bankData)
head(bankData)
tail(bankData)


#convert data from characters into usable factors
#bankData[[2]] <- as.factor(bankData[[2]])

#with loops
factor_indices <- c(2:5, 7:9, 11, 16:17)

for (i in factor_indices){
  bankData[[i]] <- as.factor(bankData[[i]])
}
head(bankData)

#find and repalce n/a
#ask R
missing <- is.na(bankData$education)
count <- sum(missing)
total <- nrow(bankData)
percent <-round((100 * count)/total,2)

#for all cols na
allMissing <- is.na(bankData)
counts <- colSums(allMissing)
total <-round((counts)/nrow(bankData)*100,2)
print(total)


#calculate averages for age duration and pdays

avgAge <- mean(bankData$age)
round(avgAge)

avgDuration <- mean(bankData$duration)
round(avgDuration)
avgPdays <- mean(bankData$pdays)
round(avgPdays)


#watch out pdays has -1 for client not previously contacted -1 is not useful for mean
#repalce -1 with na then tell R to ignore na

#option 1
fix1 <- bankData

#get the index of every row with a value of -1 [True, False,...] etc.
indices <-(fix1$pdays == -1)

countNoContact <- sum(indices)
percent <- (100 * countNoContact)/nrow(bankData)
print(paste("Percent NAS: ", round(percent), ""))

#set the value of pdays to NA on each row that had a value of -1
fix1$pdays[indices] <- NA

#exclude NAs when calculating the mean
avgPdays <- mean(fix1$pdays, na.rm = T)

print(paste("Avg Pdays", round(avgPdays), ""))

#-1 is meaningful for other stuff so dont get rid of it just ignore it for mean

#option 2

#get the index of every customer who was contacted (i.e not -1)
indices <-(bankData$pdays != -1)
head(indices)

# sum the values to get the total number contacted (T=1, F=0)
countContacted <- sum(indices)
countContacted

#only select those who have been contacted to send to our mean() function
avgPdays <- mean(bankData$pdays[indices])
round(avgPdays)

#omitting rows containing NA
no.nas <- na.omit(bankData)
percentage.left <- nrow(no.nas) / nrow(bankData) * 100
round(100 - percentage.left, 2)

#summary stats is useful to understand the shape of the data
#mean sensitive to outliers so median can show skewness
random.data <- c(14, 15, 9, 14, 8, 9, 6, 7, 8, 12)
sorted.data <- sort(random.data)
sorted.data

percentile.index <- ceiling (length(sorted.data) * 0.1)
tenth.percentile <- sorted.data[[percentile.index]]


tenth.percentile

for(i in c(0.3, 0.5, 0.75)){
  index <- ceiling(length(sorted.data) * i)
  value <- sorted.data[[index]]
  print(paste0("The ", i * 100, "th percentile is ", value))
}

#quantiles and interpolation (uses some linear interpolation method of curve fitting)
for(i in c(0.1, 0.3, 0.5, 0.75)){
  print(paste0("The ", i * 100, "th percentile is ", quantile(sorted.data, i)))
}

for(i in c(0.1, 0.3, 0.5, 0.75)){
  print(paste0("The ", i * 100, "th percentile is ", quantile(sorted.data, i, type =1)))
}

#five number summary
fivenum(bankData$age)



#trim helps us stop outliers from affecting our mean
raw.mean <- mean(bankData$balance)
raw.median <- median(bankData$balance)

#for T=0.1 we want to remove the top and bottom 10% of the values (in terms of size)
# first calcualte the tenth and ninetieth percentile

tenth.percentile <- quantile(bankData$balance, 0.1)
ninetieth.percentile <- quantile(bankData$balance, 0.9)

#then select only rows where balance is between these two values
trimmed <- bankData[bankData$balance > tenth.percentile & bankData$balance < ninetieth.percentile,]
trimmed.mean <- mean(trimmed$balance)
trimmed.mean

# the meadian value is exactly in the centre of the dataset. The smaller the difference between the mean and the meadian the more representative that mean is
raw.difference <- abs(raw.meadian - raw.mean)
trimmed.difference <- abs(raw.median - trimmed.mean)

print(paste("raw difference is", raw.difference))
print(paste("trimmed difference is", trimmed.difference))
