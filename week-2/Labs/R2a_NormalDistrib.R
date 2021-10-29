#****************************************************************************#
# (A) Follow this the steps in this file to get familiar with functions relating
#     to the normal distribution
#
# (B) After you complete all the given steps
#     (1) Write some R code to find the z-values required for a 90% confidence interval.
#     (2) Correctly state the confidence interval for a variable with SD=5cm and mean=1m.
#
# (C) Next, write some code that will find the z value corresponding to a sample
#     mean of 113cm calculated from a sample of 100 measurements. If the 
#     hypothesised mean is 1m and known SD=5cm, state the alternative hypothesis
#     and write some code to test the hypothesised against the measured mean at
#     a significance level of 1%. State the result of the test in a full sentence.
#
#****************************************************************************#

# plots for mean temperature 22, standard deviation 1.5, sample size 5
#========================================================================== 
# create a vector of x-axis values to use for plotting 
xvals<-seq(16, 28, by=0.1)

# create the values of the sampling distribution (the function dnorm does this)
fx.samp<-dnorm(xvals,22,1.5/sqrt(5))

# plot the sampling distribution
plot(xvals, fx.samp, type="l", lty=2, lwd=2, xlab="", ylab="")

# add the x-axis
abline(h=0, col="gray")

# now create the values of the normal distribution of temperatures
fx<-dnorm(xvals,22,1.5)

# and plot it - remember, lines() and points() are used to add to an already existing plot, while plot() restarts the picture
lines(xvals, fx, lwd=2)

# we add a legend to explain what was plotted (use '?legend' at the R command line to get information about all the different arguments)
legend("topright", legend=c("raw observation distribution", "sampling distribution"), lty=1:2, lwd=c(2,2), bty="n")

# now that we have the plots, let's ask probability questions 
#================================================================
# with pnorm() we can find out what the probability is of values in a distribution falling under a certain value, for example, in the normal distribution of temperatures above, what is the probability that a measured temperature will fall under 21.5
pnorm(21.5, mean=22, sd=1.5)

# or we can ask the same question for the sampling distribution (we expect to get a much smaller number here, as the sampling distribution is more 'huddled' around the mean, with fewer values in the tails)
pnorm(21.5, mean=22, sd=1.5/sqrt(5))

# note that calling pnorm() corresponds to manually looking up the normal distribution probability table, then de-normalizing the looked-up value with our mean and standard deviation


# illustrating the probabilities
#============================================
# draw the vertical boundary for the probability area for X<21.5
abline(v=21.5, col="gray")

# we manually define the polygon: it will include points only up to x=21.5
# included x-values
xvals.sub<-xvals[xvals<=21.5]
# included y-values in the instance distribution
fx.sub<-fx[xvals<=21.5]
# included y-values in the sampling distribution
fx.samp.sub<-fx.samp[xvals<=21.5]

# we add the point (0, 21.5) to the points in the distribution curve, bind them into a matrix with cbind(), then fill in the polygon, for both the instance distribution and the sampling distribution, specifying different hatching lines for the two (angle=120 vs default; lty=2 vs default)
polygon(cbind(c(21.5, xvals.sub), c(0, fx.sub)), density = 10)
polygon(cbind(c(21.5, xvals.sub), c(0, fx.samp.sub)), density = 10, angle=120, lty=2)

