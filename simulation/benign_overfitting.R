library(doParallel)
library(foreach)
library(MASS)
library(pracma)

source('simulation/toolkits.R')

parallel_run_simulations <- function(mu_array, lambda_array, n_array, p_array, snr_array, test_n, seed, chunk_size, n_iterations, output_file) {
  start_time <- Sys.time()
  cl <- makeCluster(detectCores(), outfile='')
  registerDoParallel(cl)
  functions_to_export <- c('simulate_risks', 'solve_beta_hat', 'calculate_MSE', 
                           'compute_Y', 'compute_X', 'scale_norm', 'check_orthonormal')
  clusterExport(cl, functions_to_export)
  param_list_raw <- data.frame(expand.grid(lambda=lambda_array, mu=mu_array, p=p_array, n=n_array, snr=snr_array, test_n=test_n, seed=seed, rep=1:n_iterations))
  cat('total iterations', nrow(param_list_raw))
  param_list <- param_list_raw[sample(1:nrow(param_list_raw)), ]
  # print(param_list)
  param_list <- param_list[sample(nrow(param_list)),]
  iterations <- nrow(param_list)
  num_chunks <- ceiling(iterations / chunk_size)
  for (chunk_idx in seq_len(num_chunks)) {
    start_idx <- (chunk_idx - 1) * chunk_size + 1
    end_idx <- min(chunk_idx * chunk_size, iterations)
    chunk_range <- start_idx:end_idx
    cat('start running')

    results <- foreach(idx=chunk_range, .combine=rbind, .packages=c('pracma', 'MASS')) %dopar% {
      params <- param_list[idx,]
      cat(idx/iterations*100, '%', '\n')
      mse_result <- simulate_risks(params$lambda, params$mu, params$p, params$n, params$snr, params$test_n, params$seed)
      return(data.frame(lambda=params$lambda, mu=params$mu, p=params$p, n=params$n, snr=params$snr, MSE=mse_result))
    }

    cat(paste0('Finished chunk ', chunk_idx, ' out of ', num_chunks, '\n'))
    cat('time_taken', Sys.time() - start_time)
    
    write.table(results, file=output_file, append=(chunk_idx != 1), sep=",", col.names=(chunk_idx == 1), row.names = FALSE)
  }
  
  stopCluster(cl)
}

# Main script here
time <- Sys.time()
date <- Sys.Date()

mu_array <- c(1, 100, 200, 500)
lambda_array <- c(1)
n1 <- 30
n2 <- 60
gamma_array <- generate_symlog_points(n1, n2, 0.1, 50, 1)
n_array <- c(100)
p_array <- as.integer(gamma_array * n_array)
snr_array <- seq(1, 5, length=4)
seed <- 1023
chunk_size <- 500
test_n <- 200
n_iterations <- 50

output_file <- paste0('results/R/results[', time, ']', '-', seed, '.csv')

MSE_dataframe <- parallel_run_simulations(mu_array, lambda_array, n_array, p_array, 
                          snr_array, test_n, seed, chunk_size, n_iterations, output_file)

end_time <- Sys.time()
cat('time_taken', end_time - time)

cat('Finished Running Simulations')
