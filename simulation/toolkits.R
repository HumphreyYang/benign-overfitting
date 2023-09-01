library(pracma)
library(doParallel)
library(foreach)
library(MASS)

# Function Definitions
solve_beta_hat <- function(X, Y) {
  XTX <- t(X) %*% X
  beta_hat <- ginv(XTX) %*% t(X) %*% Y
  return(beta_hat)
}

calculate_MSE <- function(beta_hat, X, Y) {
  pred_diff <- Y - X %*% beta_hat
  return(sum(pred_diff ^ 2) / length(Y))
}

compute_Y <- function(X, beta, sigma, seed) {
  if (!missing(seed)) 
    set.seed(seed) 
  epsilon <- rnorm(nrow(X), 0, sigma)
  return(X %*% beta + epsilon)
}

compute_X <- function(lambda, mu, p, n, seed) {
  C <- compute_C(lambda, p, seed)
  Gamma <- compute_Gamma(mu, n, seed)
  if (!missing(seed)) 
    set.seed(seed) 
  Z <- matrix(rnorm(p * n), p, n)
  return(C %*% Z %*% Gamma)
}

compute_C <- function(lambda, p, seed) {
  if (!missing(seed)) 
    set.seed(seed) 
  U <- randortho(p, type = 'orthonormal')
  Lambda <- diag(c(lambda, rep(1, p - 1)))
  return(U %*% Lambda %*% t(U))
}

compute_Gamma <- function(mu, n, seed) {
  if (!missing(seed)) 
    set.seed(seed) 
  V <- randortho(n, type = 'orthonormal')
  A <- diag(c(mu, rep(1, n - 1)))
  return(V %*% A %*% t(V))
}

scale_norm <- function(X, out_norm) {
  norm_X <- norm(X, 'F')
  X_normalized <- (out_norm / norm_X) * X
  print(norm_X)
  return(X_normalized)
}

# Main function
simulate_test_MSE <- function(lambda, mu, p, n, snr, seed) {
  if (!missing(seed)) 
    set.seed(seed) 
  start_time <- Sys.time()
  X <- compute_X(lambda, mu, p, n, seed)
  X <- t(X)
  train_size <- as.integer(0.7 * n)
  X_train <- X[1:train_size,]
  X_test <- X[(train_size + 1):n,]
  
  beta <- scale_norm(matrix(rep(1, p)), snr)
  sigma <- 1.0
  Y <- compute_Y(X, beta, sigma, seed)
  Y_train <- Y[1:train_size]
  Y_test <- Y[(train_size + 1):n]
  beta_hat <- solve_beta_hat(X_train, Y_train)
  
  cat(paste0('\n', strrep('*', 80), '\n'),
        paste('summary of parameters: lambda=', lambda, ', mu=', mu, ', p=', p, ', n=', n), 
        paste('summary of shapes: X shape=', dim(X), ', Y shape=', length(Y), 
            ', X_train shape=', dim(X_train), ', X_test shape=', dim(X_test), 
            ', beta_hat shape=', dim(beta_hat), ', norm_beta_hat=', norm(beta_hat, 'F'),
            ', norm_beta=', norm(beta, 'F')),
        paste('time taken = ', as.numeric(difftime(Sys.time(), start_time, units = 'secs')), ' seconds'),
        paste0('\n', strrep('*', 80), '\n'))
  
  return(calculate_MSE(beta_hat, X_test, Y_test))
}