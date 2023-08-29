library(doParallel)
library(foreach)
library(pracma)

source('toolkits.R')
parallel_run_simulations <- function(mu_array, lambda_array, n_array, p_array, snr) {
  cl <- makeCluster(detectCores(), outfile='')
  registerDoParallel(cl)
  functions_to_export <- c("simulate_test_MSE", "solve_beta_hat", "calculate_MSE", 
                           "compute_Y", "compute_X", "compute_C", "compute_Gamma", "scale_norm")
  clusterExport(cl, functions_to_export)
  param_list <- expand.grid(lambda=lambda_array, mu=mu_array, p=p_array, n=n_array, snr=snr)
  results <- foreach(idx=1:nrow(param_list), .combine=rbind, .packages='pracma') %dopar% {
    params <- param_list[idx,]
    result <- simulate_test_MSE(params$lambda, params$mu, params$p, params$n, params$snr, seed=0)
    return(result)
  }
  
  dim(results) <- c(length(mu_array), length(lambda_array), length(n_array), length(p_array))
  return(results)
}


mu_param <- seq(1, 100, length=20)
lambda_param <- seq(1, 100, length=20)
gamma_param <- seq(0.5, 50, length=20)
n_param <- c(200)
snr_param <- 5

mu_array <- mu_param
lambda_array <- lambda_param
gamma <- gamma_param
n_array <- as.integer(n_param)
p_array <- as.integer(gamma * n_array)
snr <- snr_param

MSE_matrix <- parallel_run_simulations(mu_array, lambda_array, n_array, p_array, snr)
saveRDS(MSE_matrix, "MSE_matrix.rds")

cat('Finished Running Simulations')

