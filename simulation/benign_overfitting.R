#install.packages(c("pracma", "foreach", "doParallel"))

# Required Libraries
library(pracma)
library(parallel)
library(snow)


# Solves the least squares problem using the Moore-Penrose pseudoinverse
solve_beta_hat <- function(X, Y) {
  XTX <- t(X) %*% X
  solve(XTX) %*% t(X) %*% Y
}

# Calculates the mean squared error of the prediction
calculate_MSE <- function(beta_hat, beta, X_test) {
  pred_diff <- X_test %*% beta_hat - X_test %*% beta
  sum(pred_diff ^ 2) / nrow(X_test)
}

# Computes the response variable Y
compute_Y <- function(X, beta, epsilon) {
  X %*% beta + epsilon
}

# Scale beta to have a given squared l2 norm
scale_norm <- function(beta, snr) {
  if (norm(beta, "F") == 0) return(beta)
  norm_X <- norm(beta, "F")^2
  sqrt(snr / norm_X) * beta
}

# Generate random orthonormal matrix of size dim x dim
generate_orthonormal_matrix <- function(dim) {
  qr.Q(qr(matrix(rnorm(dim * dim), nrow = dim)))
}

# Generate X = Γ Z C
compute_X <- function(lambda, mu, n, p) {
  U <- generate_orthonormal_matrix(p)
  V <- generate_orthonormal_matrix(n)
  Lambda <- diag(c(lambda, rep(1, p - 1)))
  C <- U %*% Lambda %*% t(U)
  A <- diag(c(mu, rep(1, n - 1)))
  Gamma <- V %*% A %*% t(V)
  Z <- matrix(rnorm(n * p), nrow = n)
  Gamma %*% (Z %*% C)
}

# Generate ε = N(0, σ^2 I_n)
compute_epsilon <- function(sigma, n) {
  rnorm(n, 0, sigma)
}

# Fit the LS model and calculate the test MSE
simulate_risks <- function(X, epsilon, p, n, snr, lambda, mu) {
  X_p <- X[, 1:p, drop=FALSE]
  beta <- scale_norm(matrix(rnorm(p), ncol=1), snr)
  Y <- compute_Y(X_p, beta, epsilon)
  X_train <- X_p[1:n, , drop=FALSE]
  X_test <- X_p[(n + 1):(nrow(X_p)), , drop=FALSE]
  Y_train <- Y[1:n]
  
  beta_hat <- solve_beta_hat(X_train, Y_train)
  test_MSE <- calculate_MSE(beta_hat, beta, X_test)
  cat('lambda:', lambda, 'mu:', mu, 'p:', p, 'n:', n, 'snr:', snr, 'test_MSE:', test_MSE, '\n')
  c(lambda, mu, p, n, snr, test_MSE)
}

# Simulate the test MSE for different values of λ, μ, n, p, snr
efficient_simulation <- function(mu_array, lambda_array, n_array, p_array, snr_array, sigma) {
  cl <- makeSOCKcluster(2, outfile='')
  
  # Exporting functions and libraries to all nodes
  clusterExport(cl, c("simulate_risks", "solve_beta_hat", "calculate_MSE", 
                      "compute_Y", "compute_X", "scale_norm", 
                      "generate_orthonormal_matrix", "compute_epsilon"))
  clusterEvalQ(cl, library(pracma))
  
  n <- max(n_array)
  p_max <- max(p_array)
  test_n <- 10000
  epsilon <- compute_epsilon(sigma, n + test_n)
  
  # Create parameter combinations
  params_list <- expand.grid(lambda = lambda_array,
                             mu = mu_array,
                             snr = snr_array,
                             p = p_array)
  
  # Function to perform simulation for each parameter set
  sim_func <- function(params) {
    lambda <- params['lambda']
    mu <- params['mu']
    snr <- params['snr']
    p <- params['p']
    X <- compute_X(lambda, mu, n + test_n, p_max)
    simulate_risks(X, epsilon, p, n, snr, lambda, mu)
  }
  
  cat('start running')
  # Run parallel simulation
  result_list <- do.call("rbind", parLapply(cl, split(params_list, 1:nrow(params_list)), sim_func))
  
  stopCluster(cl)
  
  return(as.data.frame(result_list))
} 

# Generate a list of points in a symmetric logarithmic scale
generate_symlog_points <- function(n1, n2, L, U, a) {
  log_part_lower <- exp(seq(log(L), log(a - 0.001), length.out = n1))
  log_part_upper <- exp(seq(log(a + 0.001), log(U), length.out = n2))
  c(log_part_lower, log_part_upper)
}


# Main Script
mu_array <- c(1)
lambda_array <- c(1)

n1 <- 30
n2 <- 30
gamma <- generate_symlog_points(n1, n2, 0.1, 10, 1)

n_array <- c(200)
p_array <- unique(round(gamma * n_array))

snr_array <- seq(1, 5, length.out = 1)
sigma <- 1

start_time <- Sys.time()

results <- efficient_simulation(mu_array, lambda_array, n_array, p_array, snr_array, sigma)
colnames(results) <- c("lambda", "mu", "p", "n", "snr", "MSE")

end_time <- Sys.time()

write.csv(results, paste("results/R/results_R_", format(start_time, "%Y%m%d_%H%M%S"), ".csv"), row.names = FALSE)

print(paste('time taken:', end_time - start_time))
print("Finished Running Simulations")