#ex 1
#calculate chi-squared value to answer the question "are there differences in
#the way that males and females vote in the UK
voters <- data.frame(
  sex = c("male","female"),
  conservative = c(313, 344),
  liberal_democrat = c(124, 158),
  labour = c(391, 388)
)
voters
barplot(c(voters$conservative, voters$liberal_democrat, voters$labour), legend=voters$sex)

# X^2 = Sum(i=0, k)(O_i - E_i)^2) / E_i
#where E_i = (n_r(i)n_c(i)) / n

#Male_n_r = 313+124+391 = 828
#Female_n_r = 344+158+388 = 890
#Conservative_n_c = 313+344 = 657
#LiberalDemocrat_n_c = 124+158= 282
#Labor_n_c = 391+388=779
#n = 313+124+391+344+158+388 = 1718

#conservative_male = (828*657)/1718 = 316.64
#etc...
#       conserv libdems labour
#male   0.04    1.04    0.64
#female 0.04    0.97    0.60

#X^2 = 0.04+1.04+0.64+0.04+0.97+0.60 = 3.33
#deg_of_freedom = (nr-1)(nc-1) = (2-1)(3-1) = 2
#look up X2 percentage points table
#v=2, p(%) upper tail = 90, = 4.605

chisq.test(voters$conservative, voters$liberal_democrat, voters$sex, correct=FALSE)

#ex 2
students <- data.frame(
  student = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  maths = c(64, 45, 40, 55, 60, 50, 80, 30, 70, 65),
  physics = c(60, 60, 55, 70, 80, 40, 85, 50, 70, 80)
)
students
stats::cor.test(students$maths, students$physics, method='pearson')
#Output:
#Pearson's product-moment correlation
#data:  students$maths and students$physics
#t = 3.132, df = 8, p-value = 0.01397
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
#  0.2112502 0.9349164
#sample estimates:
#  cor 
#0.7421625
# we now look up 0.74

#Look up 0.74 in table and under n=10 read values. 
#5% value is .632 and 1% is 0.765 so our pearsons value is 
#significant at 5% level but not 1% level.


mothers <- data.frame(
  mother = c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
  haemoglobin_x = c(11.7, 14.2, 13.7, 13.5, 14.6, 13.8, 13.9, 11.4, 11.6, 13.6),
  red_blood_cells_y = c(349, 449, 454, 441, 468, 476, 473, 448, 397, 496)
)

cor.test(mothers$haemoglobin_x, mothers$red_blood_cells_y, method = "kendall")
#Kendall's rank correlation tau
#data:  mothers$haemoglobin_x and mothers$red_blood_cells_y
#T = 30, p-value = 0.2164
#alternative hypothesis: true tau is not equal to 0
#sample estimates:
#      tau 
#0.3333333 or 1/3
#we look up a table for kendall and find that 0.33 is below
#both the 5% and the 1% significance levels.