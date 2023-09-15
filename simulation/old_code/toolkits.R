library(MASS)

library(pracma)
library(doParallel)
library(foreach)
library(MASS)

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
  Z <- matrix(rnorm(p * n), nrow = n, ncol = p)
  return(Gamma %*% Z %*% C)
}

scale_norm <- function(X, out_norm) {
  norm_X <- norm(X, 'F')**2
  return(sqrt(out_norm / norm_X) * X)
}

# Main Function
simulate_risks <- function(lambda, mu, p, n, snr, test_n, seed = NULL) {
  total_n <- n + test_n
  if (!is.null(seed)) set.seed(seed+1)
  U <- randortho(p, type = 'orthonormal')
  if (!is.null(seed)) set.seed(seed+2)
  V <- randortho(total_n, type = 'orthonormal')
  stopifnot(check_orthonormal(U), check_orthonormal(V))

  X <- compute_X(lambda, mu, p, total_n, U, V, seed)
  X_p <- X[, 1:p, drop=FALSE]
  beta <- scale_norm(matrix(rep(1, p), p, 1), snr)
  
  Y <- compute_Y(X_p, beta, 1, seed)
  null_risk <- sum((Y - mean(Y))^2) / length(Y)
  
  X_train <- X_p[1:n, , drop=FALSE]
  X_test <- X_p[(n + 1):(nrow(X)), , drop=FALSE]
  Y_train <- Y[1:n]
  Y_test <- Y[(n + 1):length(Y)]
  
  beta_hat <- solve_beta_hat(X_train, Y_train)
  test_MSE <- calculate_MSE(beta_hat, X_test, Y_test)
  
  return(test_MSE)
}