# for increasing sample size and bucket count, create histogram
#---------------------
for (i in 1:5 ) {
  j = 4 ^ i
  # name file
  png(file = paste("freq_hist_", i, ".png", sep=""))
  
  hist(rnorm(n = j * 10, mean = 50, sd = 10), breaks = seq(0, 100, length = j + 1), main=paste("n=", j * 10, ", " , j, " bins"), xlab = "")
  
  # save
  dev.off()
}

# for the final histogram, create histograms both with frequencies and with densities
#---------------------
i = 6
j = 4 ^ i
final_data = rnorm(n = j * 10, mean = 50, sd = 10)

# name file
png(file = paste("freq_hist_", i, ".png", sep=""))
hist(final_data, breaks = seq(0, 100, length = j + 1), main=paste("n=", j * 10, ", " , j, " bins"), xlab = "")
dev.off()

# name file
png(file = paste("dens_hist_", i, ".png", sep=""))
hist(final_data, breaks = seq(0, 100, length = j + 1), main=paste("n=", j * 10, ", " , j, " bins"), xlab = "", freq = FALSE)
dev.off()

# plot normal distribution
#--------------------------------
x = seq(0, 100, length = j + 1)
normal_data = dnorm(x, mean = 50, sd = 10, log = FALSE)
png(file = "pdf_norm.png")
plot(x, normal_data, main=paste("Probability density function"), xlab = "", ylab = "Probability density")
dev.off()
