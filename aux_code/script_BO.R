# Load necessary libraries
library(ggplot2)
library(viridis)
library(cowplot)
library(latex2exp)  # To use LaTeX notation in labels
library(rstudioapi)

# Set the working directory to the directory of this script

# This script calculates the minimum problem n_rep times using a GP without noise

set.seed(3)

# Define the kernel function
kernel <- function(x, y, l = 2.0, s = 1.0) {
    x <- x / l
    y <- y / l
    D <- as.matrix(dist(rbind(x, y)))[(nrow(x) + 1):(nrow(x) + nrow(y)), 1:nrow(x)]
    s * exp(-0.5 * D^2)
}

N <- 100

# Create the input grid
x <- matrix(seq(-5, 5, length = N), N, 1)

# Calculate the covariance matrix
Sigma <- kernel(x, x) + diag(N) * 1e-7
L <- t(chol(Sigma))

# Generate samples from the GP distribution
v <- rnorm(N)
z <- L %*% v

# Create the first data frame for the GP
df1 <- data.frame(x = x[,1], z = z)

# Select training points
p_sel <- sort(sample(1:100, 3))
x_train <- x[p_sel,,drop = FALSE]
f_train <- z[p_sel]

#x_train <- matrix(c(-5, -4, -3, -2, -1, c(x_train)), 8, 1)
#f_train <- c(0, 0, 0, 0, 0, f_train)


df_points <- data.frame(x = x_train[,1], y = f_train)

# Calculate the mean and covariances of the posterior distribution
x_test <- matrix(seq(-5, 5, length = N), N, 1)

mean_post <- t(kernel(x_test, x_train)) %*%
    solve(kernel(x_train, x_train) + diag(nrow(x_train)) * 1e-7) %*%
    f_train

covariances <- kernel(x_test, x_test) -
    t(kernel(x_test, x_train)) %*%
    solve(kernel(x_train, x_train) + diag(nrow(x_train)) * 1e-7) %*%
    kernel(x_test, x_train)

mean_vector <- c(mean_post)
upper <- mean_vector + sqrt(diag(covariances))
lower <- mean_vector - sqrt(diag(covariances))

plot(x_train, f_train, type = "p", col = "blue", xlim = c(-5,5), ylim = c(-1,1))
lines(x_test, mean_post, type = "l", col = "blue")
lines(x_test, upper, type = "l", col = "blue", lty = 2)
lines(x_test, lower, type = "l", col = "blue", lty = 2)


