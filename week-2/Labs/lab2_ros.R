#P2a_inference
#ex1
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
#after sampling 100 bottles mean= 197ml
# mean poured is 200ml at 1% significance
#state the null hypothesis
#h1: mean > 199
#h0: mean <= 199
# for z score  #99%	z = 2.576 from table 1%=.99547 = 0.00453
#1 sample z-test is rarly used becuase we don't know the population std dev

 hp <- 100 - 200 / 2/sqrt(100)
