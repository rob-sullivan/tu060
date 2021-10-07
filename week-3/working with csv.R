iris <- read.table("iris.csv", sep = ",", header=TRUE)
str(iris)
#convert column data type
iris <- read.table("iris.csv", sep = ",", header=TRUE, stringsAsFactors = TRUE)
str(iris)
#replace empty with nulls
iris <- read.table("iris.csv", sep = ",", header=TRUE, stringsAsFactors = TRUE, fill = TRUE)
#fix missing headings
iris <- read.table("iris.csv", sep = ",", header=TRUE, stringsAsFactors = TRUE, fill = TRUE, na.strings = c("", "?", "NA"))
#readr is x10 times faster, it produces tribbles like data frames but they tweak some older behavious.
library(readr)
iris <- read_csv("iris.csv")
print(iris)
problems(iris)
#read in specific columns
iris <- read_csv("iris.csv", col_types = "dd?_dc")
print(iris)
#can save data as RData format which is binary, faster and can store more than text files
load("pokemon.RData")
readRDS('pokemon.rds')
# can also use readr
read_rds("pokemon.rds")
