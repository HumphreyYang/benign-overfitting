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
    params <- param_list[idx,]
    mse_result <- simulate_test_MSE(params$lambda, params$mu, params$p, params$n, params$snr, params$seed)
    return(data.frame(lambda=params$lambda, mu=params$mu, p=params$p, n=params$n, snr=params$snr, MSE=mse_result))
  }
  
  stopCluster(cl)
  return(results)
}

parallel_run_simulations_in_chunks <- function(mu_array, lambda_array, n_array, p_array, snr_array, seed, chunk_size, output_file) {
  start_time = Sys.time()
  cat('Starting Run')
  cl <- makeCluster(detectCores(), outfile='')
  registerDoParallel(cl)
  source('simulation/toolkits.R')
  functions_to_export <- c('simulate_test_MSE', 'solve_beta_hat', 'calculate_MSE', 
                           'compute_Y', 'compute_X', 'scale_norm', 'check_orthonormal')
  clusterExport(cl, functions_to_export)
  param_list <- expand.grid(lambda=lambda_array, mu=mu_array, p=p_array, n=n_array, snr=snr_array, seed=seed)
  param_list <- param_list[sample(nrow(param_list)),]
  iterations <- nrow(param_list)
  
  num_chunks <- ceiling(iterations / chunk_size)
  for (chunk_idx in seq_len(num_chunks)) {
    start_idx <- (chunk_idx - 1) * chunk_size + 1
    end_idx <- min(chunk_idx * chunk_size, iterations)
    chunk_range <- start_idx:end_idx
    
    results <- foreach(idx=chunk_range, .combine=rbind, .packages=c('pracma', 'MASS')) %dopar% {
      params <- param_list[idx,]
      mse_result <- simulate_test_MSE(params$lambda, params$mu, params$p, params$n, params$snr, params$seed)
      return(data.frame(lambda=params$lambda, mu=params$mu, p=params$p, n=params$n, snr=params$snr, MSE=mse_result))
    }
    
    cat(paste0('Finished chunk ', chunk_idx, ' out of ', num_chunks, '\n'))
    cat('time_taken', Sys.time() - start_time)
    if (chunk_idx == 1) {
      write.csv(results, output_file, row.names = FALSE)
    } else {
      write.table(results, file=output_file, append=TRUE, sep=",", col.names=FALSE, row.names = FALSE)
    }
  }
  
  stopCluster(cl)
}


time = Sys.time()
date = Sys.Date()

mu_array <- seq(1, 100, length=8)
lambda_array <- c(1)
gamma_array <- seq(0.1, 10, length=100)
n_array <- c(200)
p_array <- as.integer(gamma_array * n_array)
snr_array <- seq(1, 5, length=4)
seed <- 1023
chunk_size <- 80000

MSE_dataframe <- parallel_run_simulations_in_chunks(mu_array, lambda_array, n_array, p_array, 
                          snr_array, seed, chunk_size, output_file=paste0('results/results[', time, ']', '-', seed, '.csv'))
end_time <- Sys.time()
cat('time_taken', end_time - time)

cat('Finished Running Simulations')
