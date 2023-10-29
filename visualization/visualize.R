# install.packages("viridis")
library("viridis") 
library(RColorBrewer)

symlog_transform <- function(x) {
  ifelse(x >= 1, log10(x), -log10(1 / x))
}

draw_plots <- function(df, title, y_max=10, y_gap=5, lines=TRUE){
    # Initialize plot
    plot(NULL, xlim = c(-1, 1), ylim = c(0, y_max), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10))
    axis(2, at = seq(0, y_max, by = y_gap))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), col = 'grey', lty = 2)
    # Plotting the points and lines
    unique_snrs <- unique(df$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        if (lines){
            abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        }
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topright', legend = snr_legend, col = colors, pch = 19, cex=2)
    if (title == 'lambda'){
        title(main=bquote(~ lambda == .(round(lambda_val, 2))), cex.main=2)
    } else if (title == 'mu'){
        title(main=bquote(~ mu == .(round(mu_val, 2))), cex.main=2)
    } else if (title == 'rho'){
        title(main=bquote(~ rho == .(round(rho_val, 2))), cex.main=2)
    }

    if("true_p" %in% colnames(df)){
        abline(v = symlog_transform(unique(df$true_p / df$n)), col = 'brown', lwd = 2)
    }
}

file_name <- 'results/Python/lambda_muresults_[04-10-2023_16:55:21-1655].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]
df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2), ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu')
    # Close the PNG device
    dev.off()
}


file_name <- 'results/Python/lambda_muresults_[30-09-2023_22:32:06-1239].csv'
# file_name <- 'results/Python/lambda_muresults_[04-10-2023_18:58:28-1858].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', 
    round(lambda_val, 2), ".png"), width = 800, height = 800)
    draw_plots(df_lambda, 'lambda')
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_quad_results_[19-10-2023_16:20:49-2025].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]

df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
    
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2),'_quad', ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu')

    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_quad_results_[19-10-2023_16:27:50-2025].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), '_quad', ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda')
    
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_abs_results_[19-10-2023_16:06:31-2025].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]

df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
    
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2),'_abs', ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu')

    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_abs_results_[19-10-2023_16:13:40-2025].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), '_abs', ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda')
    
    # Close the PNG device
    dev.off()
}


file_name <- 'results/Python/lambda_muresults_[30-09-2023_22:58:04-1239].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), "_expend", ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda')
    
    # Close the PNG device
    dev.off()
}


file_name <- 'results/Python/compoundresults_[21-09-2023_11:19:07-1].csv'
df <- read.csv(file_name)
colnames(df) <- c('rho', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (rho_val in unique(df$rho)){
    df_rho = subset(df, rho == rho_val)

    # Open a PNG device
    png(paste0("visualization/figures/", expression(rho), '_', round(rho_val, 2), ".png"), width = 800, height = 800)

    draw_plots(df_rho, 'rho')
    # Close the PNG device
    dev.off()
}



file_name <- 'results/Python/compound_randomresults_[21-09-2023_11:47:41-1].csv'
df <- read.csv(file_name)
colnames(df) <- c('rho', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (rho_val in unique(df$rho)){
    df_rho = subset(df, rho == rho_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", 'random', expression(rho), '_', round(rho_val, 2), ".png"), width = 800, height = 800)
    draw_plots(df_rho, 'rho')
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_bias_linear_results_[24-10-2023_10:17:53-1655].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]
df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2), '_bias', ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu', lines=TRUE)
    # Close the PNG device
    dev.off()
}


file_name <- 'results/Python/lambda_mu_bias_linear_results_[24-10-2023_10:26:16-1858].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), "_bias", ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda', y_max=500, y_gap=100, lines=FALSE)
    
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_bias_linear_100_results_[29-10-2023_16:02:23-1655].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'true_p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]
df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2), '_bias_tp_100', ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu', lines=TRUE)
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_bias_linear_100_results_[29-10-2023_17:51:10-1858].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'true_p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), "_bias_tp_100", ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda', lines=FALSE)
    
    # Close the PNG device
    dev.off()
}


file_name <- 'results/Python/lambda_mu_bias_linear_400_results_[29-10-2023_20:39:05-1655].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'true_p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]
df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2), '_bias_tp_400', ".png"), width = 800, height = 800)

    draw_plots(df_mu, 'mu', lines=TRUE)
    # Close the PNG device
    dev.off()
}

file_name <- 'results/Python/lambda_mu_bias_linear_400_results_[29-10-2023_20:39:09-1858].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'true_p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), "_bias_tp_400", ".png"), width = 800, height = 800)
    
    draw_plots(df_lambda, 'lambda', lines=FALSE)
    
    # Close the PNG device
    dev.off()
}