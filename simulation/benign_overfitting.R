library(foreach)
library(doParallel)
library(matrixStats)

solve_beta_hat <- function(X, Y) {
  XTX <- t(X) %*% X
  solve(XTX) %*% t(X) %*% Y
}

calculate_MSE <- function(beta_hat, X, Y) {
  pred_diff <- Y - X %*% beta_hat
  sum(pred_diff^2) / nrow(X)
}

compute_Y <- function(X, beta, epsilon) {
  X %*% beta + epsilon
}

scale_norm <- function(beta, snr) {
  if (norm(beta, "2") == 0) return(beta)
  norm_X <- norm(beta, "2")^2
  sqrt(snr / norm_X) * beta
}

generate_orthonormal_matrix <- function(dim, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  qr.Q(qr(matrix(rnorm(dim * dim), nrow=dim)))
}

compute_X <- function(lambda, mu, n, p, seed=NULL) {
  U <- generate_orthonormal_matrix(p, seed)
  V <- generate_orthonormal_matrix(n, seed)
  Lambda <- diag(c(lambda, rep(1, p - 1)))
  A <- diag(c(mu, rep(1, n - 1)))
  C <- U %*% Lambda %*% t(U)
  Gamma <- V %*% A %*% t(V)
  if (!is.null(seed)) set.seed(seed)
  Z <- matrix(rnorm(n * p), nrow=n)
  Gamma %*% (Z %*% C)
}

compute_epsilon <- function(sigma, n, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  rnorm(n, mean=0, sd=sigma)
}

simulate_risks <- function(X, epsilon, params) {
  lambda <- params[1]
  mu <- params[2]
  p <- params[3]
  n <- params[4]
  snr <- params[5]
  
  X_p <- X[, 1:p, drop=FALSE]
  beta <- scale_norm(rep(1, p), snr)
  print(norm(beta, "2")^2)
  
  Y <- compute_Y(X_p, beta, epsilon)
  null_risk <- sum((Y - mean(Y))^2) / length(Y)
  
  X_train <- X_p[1:n, , drop=FALSE]
  X_test <- X_p[(n + 1):(nrow(X)), , drop=FALSE]
  Y_train <- Y[1:n]
  Y_test <- Y[(n + 1):length(Y)]
  
  beta_hat <- solve_beta_hat(X_train, Y_train)
  test_MSE <- calculate_MSE(beta_hat, X_test, Y_test)
  
  c(lambda, mu, p, n, snr, test_MSE, null_risk)
}

generate_symlog_points <- function(n1, n2, L, U, a) {
  log_part_lower <- exp(seq(log(L), log(a - 0.001), length.out=n1))
  log_part_upper <- exp(seq(log(a + 0.001), log(U), length.out=n2))
  c(log_part_lower, log_part_upper)
}

# Main function for simulations
efficient_simulation <- function(mu_array, lambda_array, n_array, p_array, snr_array, sigma, seed=NULL) {
  registerDoParallel(cores=4)
  n <- max(n_array)
  max_p <- max(p_array)
  test_n <- 10000
  epsilon <- compute_epsilon(sigma, n + test_n, seed + 1)
  
  total_combinations <- length(mu_array) * length(lambda_array) * length(n_array) * length(p_array) * length(snr_array)
  result_arr <- matrix(0, nrow=total_combinations, ncol=7)
  idx <- 1
  
  foreach(lambda=lambda_array, .combine=rbind, .inorder=FALSE) %:%
    foreach(mu=mu_array, .combine=rbind, .inorder=FALSE) %:%
    foreach(snr=snr_array, .combine=rbind, .inorder=FALSE) %:%
    foreach(p=p_array, .combine=rbind, .inorder=FALSE) %dopar% {
      params <- c(lambda, mu, p, n, snr)
      X <- compute_X(lambda, mu, n + test_n, max_p, seed)
      result_arr[idx, ] <- simulate_risks(X, epsilon, params)
      idx <- idx + 1
    }
  return(result_arr)
}

# Main script
mu_array <- c(1, 100, 200, 500)
lambda_array <- c(1)
n1 <- 30
n2 <- 30
gamma <- generate_symlog_points(n1, n2, 0.1, 10, 1)
n_array <- c(200)
p_array <- unique(round(gamma * n_array))
snr_array <- seq(1, 5, length.out=4)
sigma <- 1
seed <- 2355

result_arr <- efficient_simulation(mu_array, lambda_array, n_array, p_array, snr_array, sigma, seed)
colnames(result_arr) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE', 'null_risk')
write.csv(result_arr, file=paste0('results_R_', Sys.Date(), '_', seed, '.csv'), row.names=FALSE)
