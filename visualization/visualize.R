# install.packages("viridis")
library("viridis") 
library(RColorBrewer)

file_name <- 'results/R/results[2023-09-28 00:06:42]-1023.csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n
df <- df[order(df$gamma ),]

symlog_transform <- function(x) {
  ifelse(x >= 1, log10(x), -log10(1 / x))
}

symlog_back_transform <- function(y) {
  ifelse(y >= 0, 10^y, 1 / (10^(-y)))
}

df$transformed_gamma <- symlog_transform(df$gamma)
for (mu_val in unique(df$mu)){
    df_mu = subset(df, mu == mu_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(mu), '_', round(mu_val, 2), ".png"), width = 800, height = 800)

    # Initialize plot
    plot(NULL, xlim = c(-1, 1), ylim = c(0, 10), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10))
    axis(2, at = seq(0, 10, by = 5))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), col = 'grey', lty = 2)

    # Plotting the points and lines
    unique_snrs <- unique(df_mu$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df_mu, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topright', legend = snr_legend, col = colors, pch = 19, cex=2)

    title(main=bquote(~ mu == .(round(mu_val, 2))), cex.main=2)
    # Close the PNG device
    dev.off()
}


file_name <- '/home/humphreyyang/code/benign-overfitting/results/Python/lambda_muresults_[30-09-2023_22:32:06-1239].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

print(df$gamma)

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), ".png"), width = 800, height = 800)

    # Initialize plot
    plot(NULL, xlim = c(-1, 1), ylim = c(0, 10), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10))
    axis(2, at = seq(0, 10, by = 5))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), col = 'grey', lty = 2)

    # Plotting the points and lines
    unique_snrs <- unique(df_lambda$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df_lambda, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topright', legend = snr_legend, col = colors, pch = 19, cex=2)

    title(main=bquote(~ lambda == .(round(lambda_val, 2))), cex.main=2)
    # Close the PNG device
    dev.off()
}


file_name <- '/home/humphreyyang/code/benign-overfitting/results/Python/lambda_muresults_[28-09-2023_20:13:03-1909].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

print(df$gamma)

df$transformed_gamma <- symlog_transform(df$gamma)
for (lambda_val in unique(df$lambda)){
    df_lambda = subset(df, lambda == lambda_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", expression(lambda), '_', round(lambda_val, 2), "expend", ".png"), width = 800, height = 800)

    # Initialize plot
    plot(NULL, xlim = c(-1, 1.75), ylim = c(0, 10), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50))
    axis(2, at = seq(0, 10, by = 5))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10, 10, 20, 50)), col = 'grey', lty = 2)

    # Plotting the points and lines
    unique_snrs <- unique(df_lambda$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df_lambda, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topleft', legend = snr_legend, col = colors, pch = 19, cex=2)

    title(main=bquote(~ lambda == .(round(lambda_val, 2))), cex.main=2)
    # Close the PNG device
    dev.off()
}


file_name <- '/home/humphreyyang/code/benign-overfitting/results/Python/compoundresults_[21-09-2023_11:19:07-1].csv'
df <- read.csv(file_name)
colnames(df) <- c('rho', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (rho_val in unique(df$rho)){
    df_rho = subset(df, rho == rho_val)

    # Open a PNG device
    png(paste0("visualization/figures/", expression(rho), '_', round(rho_val, 2), ".png"), width = 800, height = 800)

    # Initialize plot
    plot(NULL, xlim = c(-1, 1), ylim = c(0, 10), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10))
    axis(2, at = seq(0, 10, by = 5))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), col = 'grey', lty = 2)

    # Plotting the points and lines
    unique_snrs <- unique(df_rho$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df_rho, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topright', legend = snr_legend, col = colors, pch = 19, cex=2)

    title(main=bquote(~ rho == .(round(rho_val, 2))), cex.main=2)
    # Close the PNG device
    dev.off()
}



file_name <- '/home/humphreyyang/code/benign-overfitting/results/Python/compound_randomresults_[21-09-2023_11:47:41-1].csv'
df <- read.csv(file_name)
colnames(df) <- c('rho', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

df$transformed_gamma <- symlog_transform(df$gamma)
for (rho_val in unique(df$rho)){
    df_rho = subset(df, rho == rho_val)
      
    # Open a PNG device
    png(paste0("visualization/figures/", 'random', expression(rho), '_', round(rho_val, 2), ".png"), width = 800, height = 800)

    # Initialize plot
    plot(NULL, xlim = c(-1, 1), ylim = c(0, 10), 
        xlab = expression(gamma), ylab = 'MSE', 
        xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i', 
        cex.lab = 1.25, cex.axis = 1.25)

    # Add custom axis
    axis(1, at = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), 
        labels = c(0.1, 0.2, 0.5, 1, 2, 5, 10))
    axis(2, at = seq(0, 10, by = 5))

    # Add grid
    abline(v = symlog_transform(c(0.1, 0.2, 0.5, 1, 2, 5, 10)), col = 'grey', lty = 2)

    # Plotting the points and lines
    unique_snrs <- unique(df_rho$snr)
    colors <- palette()

    snr_legend <- numeric()

    for (i in 1:length(unique_snrs)) {
        snr_val <- unique_snrs[i]
        
        sub_df <- subset(df_rho, snr == snr_val)
        snr_legend <- c(snr_legend, paste0('SNR = ', round(snr_val, 2)))
        
        points(sub_df$transformed_gamma, sub_df$MSE, pch = 19, cex=1.5, col = colors[i])
        
        # Separate gamma into two ranges and apply linear spline interpolation
        sub_df_s <- subset(sub_df, gamma <= 1)
        sub_df_l <- subset(sub_df, gamma >= 1)
        abline(h=snr_val, col = colors[i], lty = 2, lwd = 3)
        
        for (sub_df in list(sub_df_s, sub_df_l)) {
        if (nrow(sub_df) > 1) {
            lines(symlog_transform(sub_df$gamma), sub_df$MSE, col = colors[i], lwd = 3)
        }
        }
    }

    # Add legend
    legend('topleft', legend = snr_legend, col = colors, pch = 19, cex=2)

    title(main=bquote(~ rho ~ "~UNIF(" ~.(-rho_val)~ "," ~.(rho_val)~ ")"), cex.main=2)
    # Close the PNG device
    dev.off()
}
