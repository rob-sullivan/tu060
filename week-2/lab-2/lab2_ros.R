#P2a_inference
#ex1 - confidence interval
#problem 200ml bottle
  #n observations = 100
  # mean = 200
  # standard div = 2
  # get z score for confidence level 95%
  #Confidence Interval	Z (#https://www.mathsisfun.com/data/confidence-interval.html)
  #80%	1.282
  #85%	1.440
  #90%	1.645
  #95%	1.960
  #99%	2.576
  #99.5%	2.807
  #99.9%	3.291

cl <- (1.95*(2/sqrt(100)))
print(paste("confidence is 200 +/-", cl))
print(paste("so either", 200-cl, "ml or ", 200+cl, "ml"))

#ex2 hypothesis 
#after sampling 100 bottles sample mean=197ml
# mean poured is 200ml at 1% significance
#state the null hypothesis
#h1: mean > 199
#h0: mean <= 199

#Once you get the sample statistics, you can determine the p-value 
#through different methods. 
#The most common methods are the T-score and Z-score for normal distributions.
#https://www.educba.com/z-score-vs-t-score/
# for z score x-mu/std dev #99%	z = 2.576 from table 1%=.99547 = 0.00453
#1 sample z-test is rarely used because we don't know the population std dev

hp <- 100 - 200 / 2/sqrt(100)
hp

#it's called a lower-tailed test
#if h0: mu >= 150
#if h1: mu < 150

#upper tail test
#if h0: mu <= 500k
#if h1: mu > 500k

#2-Tailed Test
#if h0: mu = 500k
#if h1: mu != 25


