library(doParallel)
library(pracma)

# for_run_simulation <- function(mu_array, lambda_array, n_array, p_array, snr, seed) {
#   source('simulation/toolkits.R')
#   functions_to_export <- c('simulate_test_MSE', 'solve_beta_hat', 'calculate_MSE', 
#                            'compute_Y', 'compute_X', 'scale_norm', 'check_orthonormal')
#   param_list <- expand.grid(lambda=lambda_array, mu=mu_array, p=p_array, n=n_array, snr=snr, seed=seed)
#   iterations <- nrow(param_list)

#   progress_print <- function(idx, iterations) cat(paste0(idx/iterations, "/1 \n"))

#   results_df <- data.frame()
#   for (idx in 1:iterations){
#     progress_print(idx, iterations)
#     params <- param_list[idx,]
#     mse_result <- simulate_test_MSE(params$lambda, params$mu, params$p, params$n, params$snr, params$seed)
    
#     # Create a data frame to hold current parameters and result
#     temp_df <- data.frame(lambda=params$lambda, mu=params$mu, p=params$p, n=params$n, snr=params$snr, MSE=mse_result)}
# }

parallel_run_simulations <- function(mu_array, lambda_array, n_array, p_array, snr, seed) {
  cl <- makeCluster(detectCores(), outfile='')
  registerDoParallel(cl)
  source('simulation/toolkits.R')
  functions_to_export <- c('simulate_test_MSE', 'solve_beta_hat', 'calculate_MSE', 
                           'compute_Y', 'compute_X', 'scale_norm', 'check_orthonormal')
  clusterExport(cl, functions_to_export)
  param_list <- expand.grid(lambda=lambda_array, mu=mu_array, p=p_array, n=n_array, snr=snr, seed=seed)
  iterations <- nrow(param_list)

  progress_print <- function(idx, iterations) cat(paste0(idx/iterations, "/1 \n"))

  results_df <- data.frame()
  results <- foreach(idx=1:iterations, .combine=rbind, .packages=c('pracma', 'MASS')) %dopar% {
    progress_print(idx, iterations)
    params <- param_list[idx,]
    mse_result <- simulate_test_MSE(params$lambda, params$mu, params$p, params$n, params$snr, params$seed)
    
    temp_df <- data.frame(lambda=params$lambda, mu=params$mu, p=params$p, n=params$n, snr=params$snr, MSE=mse_result)

    return(temp_df)
  }
  
  return(results)
}

time = Sys.time()
date = Sys.Date()

mu_array <- seq(1, 20, length=40)
lambda_array <- seq(1, 20, length=40)
gamma_array <- seq(0.05, 5.05, length=500)
n_array <- c(100)
p_array <- as.integer(gamma_array * n_array)
snr <- 5
seed <- 105

MSE_dataframe <- parallel_run_simulations(mu_array, lambda_array, n_array, p_array, snr, seed)

end_time <- Sys.time()
cat('time_taken', end_time - time)

# Save to CSV
write.csv(MSE_dataframe, paste0('results/results[', time, ']', '-', seed, '.csv'), row.names = FALSE)

cat('Finished Running Simulations')
