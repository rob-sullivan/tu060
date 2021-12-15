#code graveyard :(
```{r}
# before installing kableExtra, do this first if on Ubuntu: sudo apt -y install libfontconfig1-dev ref: https://github.com/r-lib/systemfonts/issues/35
#install.packages("kableExtra", dependencies = TRUE)  # we use this for creating custom tables
library(kableExtra)
school1_student_names_table <- data.frame(Attribute=school1_student_names, mpg_box = "", mpg_hist = "")
kbl(school1_student_names_table , caption = "GP School Descriptive Statistics Table: Students", booktabs = TRUE) %>%
  kable_paper(full_width = FALSE)
````
```{r}
mpg_list <- split(mtcars$mpg, mtcars$cyl)
disp_list <- split(mtcars$disp, mtcars$cyl)
inline_plot <- data.frame(cyl = c(4, 6, 8), mpg_box = "", mpg_hist = "", mpg_line1 = "", mpg_line2 = "", mpg_points1 = "", mpg_points2 = "", mpg_poly = "")

inline_plot %>%
  kbl(caption = "Demo Table (Landscape)[note]", booktabs = TRUE) %>%
  kable_paper(full_width = FALSE) %>%
  column_spec(2, image = spec_boxplot(mpg_list)) %>%
  column_spec(3, image = spec_hist(mpg_list)) %>%
  column_spec(4, image = spec_plot(mpg_list, same_lim = TRUE)) %>%
  column_spec(5, image = spec_plot(mpg_list, same_lim = FALSE)) %>%
  column_spec(6, image = spec_plot(mpg_list, type = "p")) %>%
  column_spec(7, image = spec_plot(mpg_list, disp_list, type = "p")) %>%
  column_spec(8, image = spec_plot(mpg_list, polymin = 5)) %>%
  add_footnote(c("This table is from mtcars",
                 "Group 1 contains mpg, cyl and disp",
                 "Group 2 contains hp, drat and wt"),
               notation = "symbol")
```

```{r}
#get data we are interested in for out boxplot
data <- data.frame(students[[1]], students[[2]], students[[6]], students[[7]])

#prepare our data
colnames(data)[1] <- "sex"
colnames(data)[2] <- "age"
colnames(data)[3] <- "Medu"
colnames(data)[4] <- "Fedu"
melt_A<-melt(data)

# grouped boxplot
ggplot(melt_A,aes(x=sex,y=value,fill=sex))+
  geom_boxplot()+
  facet_wrap(~variable)
```



```{r}
math_title <- c('traveltime.m', 'studytime.m', 
                'failures.m','famrel.m', 
                'freetime.m', 'goout.m',
                'Walc.m', ' health.m', 
                'absences.m','mG1', 
                'mG2', 'mG3')
math_skew <- c(skewness(maths$traveltime.m), skewness(maths$studytime.m), 
               skewness(maths$failures.m), skewness(maths$famrel.m),
               skewness(maths$freetime.m), skewness(maths$goout.m),
               skewness(maths$Walc.m), skewness(maths$health.m),
               skewness(maths$absences.m), skewness(maths$mG1),
               skewness(maths$mG2), skewness(maths$mG3))
math_kurtosis <- c(kurtosis(maths$traveltime.m), kurtosis(maths$studytime.m), 
                   kurtosis(maths$failures.m), kurtosis(maths$famrel.m),
                   kurtosis(maths$freetime.m), kurtosis(maths$goout.m),
                   kurtosis(maths$Walc.m), kurtosis(maths$health.m),
                   kurtosis(maths$absences.m), kurtosis(maths$mG1),
                   kurtosis(maths$mG2), kurtosis(maths$mG3))
math_comment <- vector()

for(s in math_skew){
  #1.6366852
  if(-0.5 > s & s < 0.5){
    math_comment <- append(paste(s, "fairly symmetrical", sep=" is "), math_comment)
  }
  else if((-1.0 > s) & (s < -0.5) | (0.5 > s) & (s <1.0)){
    math_comment <- append(paste(s, "moderately skewed", sep=" is "), math_comment)
  }
  else if(-1.0 < s & s > 1.0){
    math_comment <- append(paste(s, "highly skewed", sep=" is "), math_comment)
  }
}

i = 1
for(k in math_kurtosis){
  if(3.0 < k){
    math_comment[i] <- paste(math_comment[i], "leptokurtic distribution(tall+thin)", sep=" and ")
  }
  else if(3.0 > k){
    math_comment[i] <- paste(math_comment[i], "platykurtic distribution(Moderately Spread Out)", sep=" and ")
  }
  else if(k == 3.0){
    math_comment[i] <- paste(math_comment[i], "mesokurtic distribution(similar to normal distribution)", sep=" and ")
  }
  i = i + 1
}


data <- data.frame(math_title, math_skew, math_kurtosis, math_comment)
colnames(data)[1] <- "Attributes"
colnames(data)[2] <- "Skewness"
colnames(data)[3] <- "Kurtosis"
colnames(data)[4] <- "Comment"
head(data)
```


```{r}
varsint<-c("school",
           "age", 
           "Pstatus", 
           "guardian.m", 
           "traveltime.m",
           "failures.m", 
           "schoolsup.m" , 
           "romantic.m",
           "goout.m",
           "Dalc.m",
           "Walc.m",
           "health.m",
           "mG3"
)
```