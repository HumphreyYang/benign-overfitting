# install.packages("viridis")
library("viridis") 
library(RColorBrewer)

file_name <- '/home/humphreyyang/code/benign-overfitting/results/Python/results_[15-09-2023_21:52:30-1458].csv'
df <- read.csv(file_name)
colnames(df) <- c('lambda', 'mu', 'p', 'n', 'snr', 'MSE')
df$gamma <- df$p / df$n

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
    png(paste0("visualization/figures/isotropic_features", expression(mu), '_', round(mu_val, 2), ".png"), width = 800, height = 800)

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
