library(pracma)
library(doParallel)
library(foreach)
library(MASS)

# Auxiliary Functions
check_orthonormal <- function(A) {
  n <- nrow(A)
  col_norms <- sqrt(rowSums(A^2))
  ortho_check <- t(A) %*% A
  identity_matrix <- diag(n)
  return(all(abs(col_norms - 1) < 1e-8) && all(abs(ortho_check - identity_matrix) < 1e-8))
}

solve_beta_hat <- function(X, Y) {
  XTX <- t(X) %*% X
  return(ginv(XTX) %*% t(X) %*% Y)
}

calculate_MSE <- function(beta_hat, X, Y) {
  pred_diff <- Y - X %*% beta_hat
  return(sum(pred_diff^2) / length(Y))
}

compute_Y <- function(X, beta, sigma, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  epsilon <- rnorm(nrow(X), 0, sigma)
  return(X %*% beta + epsilon)
}

compute_X <- function(lambda, mu, p, n, U, V, seed = NULL) {
  C <- U %*% diag(c(lambda, rep(1, p - 1))) %*% t(U)
  Gamma <- V %*% diag(c(mu, rep(1, n - 1))) %*% t(V)
  if (!is.null(seed)) set.seed(seed)
  Z <- matrix(rnorm(p * n), nrow = p, ncol = n)
  return(C %*% Z %*% Gamma)
}

scale_norm <- function(X, out_norm) {
  norm_X <- norm(X, 'F')
  return((out_norm / norm_X) * X)
}

# Main Function
simulate_test_MSE <- function(lambda, mu, p, n, snr, seed = NULL) {
  if (!is.null(seed)) set.seed(seed+1)
  U <- randortho(p, type = 'orthonormal')
  if (!is.null(seed)) set.seed(seed+2)
  V <- randortho(n, type = 'orthonormal')
  stopifnot(check_orthonormal(U), check_orthonormal(V))

  if (!is.null(seed)) set.seed(seed)
  X <- t(compute_X(lambda, mu, p, n, U, V, seed))
  train_size <- as.integer(0.7 * n)
  X_train <- X[1:train_size, ]
  X_test <- X[(train_size + 1):n, ]
  
  beta <- scale_norm(matrix(rep(1, p), p, 1), snr)
  sigma <- 1.0
  Y <- compute_Y(X, beta, sigma, seed)
  Y_train <- Y[1:train_size]
  Y_test <- Y[(train_size + 1):n]
  
  beta_hat <- solve_beta_hat(X_train, Y_train)
  return(calculate_MSE(beta_hat, X_test, Y_test))
}