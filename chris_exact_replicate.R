# ============================================================================
# chris_exact_replicate.R
#
# Standalone replication attempt using C.K. Wikle's own example
# hyperparameters exactly as stated in Chris_model_description-v1.pdf
# Section 6 / Figure 2: p=10 EOFs, K=5 (short lags), W=36 (long window),
# D=4 (Legendre degrees), tau=6. E bumped from the paper's own example
# E=100 to E=500 to match every GRU variant's own 500-member-per-seed
# convention elsewhere in the writeup -- see dsno_fit_h6.R's header for why
# 500 independent networks, not 500 draws, is the faithful way to scale
# this model's ensemble.
#
# p=10 EOFs are just the first 10 columns of our already-computed K=27
# basis (EOFs are nested/hierarchical -- truncating further requires no
# recomputation, since EOF_k doesn't depend on how many total are kept).
#
# This script's output (chris_exact_oct2026.rds/.pdf) now IS used directly
# by the writeup (Figure 7, Table 1's Chris-exact column, Figure 1's EOF1
# analog) -- not merely diagnostic/exploratory as originally written.
# ============================================================================

source("dsno_h6.R")   # loads eof_results.RData (A_K, Psi_K, col_mean) + all dsno_h6 functions

P_CHRIS <- 10L
A_10 <- A_K[, seq_len(P_CHRIS), drop = FALSE]
Psi_10 <- Psi_K[, seq_len(P_CHRIS), drop = FALSE]

CFG <- list(K = 5L, W = 36L, D = 4L, tau = 6L, E = 500L)
cat(sprintf("Chris-exact config: p=%d, K=%d, W=%d, D=%d, E=%d\n", P_CHRIS, CFG$K, CFG$W, CFG$D, CFG$E))

B <- build_legendre_basis(CFG$W, CFG$D)
ds <- build_dual_stream(A_10, CFG$K, CFG$W, CFG$D, CFG$tau, B)
d_in <- P_CHRIS * (CFG$K + 1 + CFG$D)
cat(sprintf("Training windows: %d, d_in=%d\n", nrow(ds$X), d_in))

t0 <- Sys.time()
fits <- fit_dsno_ensemble(ds$X, ds$Y, E = CFG$E, d_in = d_in, p = P_CHRIS, seed0 = 1L,
                           epochs = 500L, patience = 20L, verbose_every = 501)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time: %.1f min\n", fit_time))

X_cur <- build_current_features(A_10, CFG$K, CFG$W, CFG$D, B)
arr_coef <- predict_dsno_ensemble_array(fits, X_cur)  # (1, p=10, E=100)
draws_pc <- arr_coef[1, , ]  # (10, 100)

pc1 <- draws_pc[1, ]
cat(sprintf("\nOct-2026 EOF1 (Chris-exact): median=%.1f, mean=%.1f, [5%%,95%%]=[%.1f,%.1f]\n",
            median(pc1), mean(pc1), quantile(pc1, 0.05), quantile(pc1, 0.95)))
cat("(compare: our tuned K=3/D=2/p=27 gave median=19.8; GRU-iterated gave median=43.4; true historical record=54.5)\n")

saveRDS(list(draws_pc = draws_pc, CFG = CFG, p = P_CHRIS), "chris_exact_oct2026.rds")

# ------------------------------------------------------------------
# Physical-space reconstruction + joint-envelope 3-panel figure (same
# method as final_plot_oct2026.R, copied inline to avoid sourcing that
# file's unrelated FNO-DST setup cost).
# ------------------------------------------------------------------
library(fields); library(RColorBrewer)
field_from_r_order <- function(values_r_order) {
  full <- rep(NA_real_, nlon * nlat); full[ocean_idx] <- values_r_order
  matrix(full, nrow = nlon, ncol = nlat)
}
smooth_grid <- function(mat, factor = 6) {
  land <- is.na(mat); filled <- mat; filled[land] <- mean(mat, na.rm = TRUE)
  fine_lon <- seq(min(lon), max(lon), length.out = length(lon) * factor)
  fine_lat <- seq(min(lat), max(lat), length.out = length(lat) * factor)
  z_fine <- interp.surface.grid(list(x = lon, y = lat, z = filled), list(x = fine_lon, y = fine_lat))$z
  mask_fine <- interp.surface.grid(list(x = lon, y = lat, z = matrix(as.numeric(!land), nlon, nlat)),
                                    list(x = fine_lon, y = fine_lat))$z
  z_fine[mask_fine < 0.5] <- NA
  list(x = fine_lon, y = fine_lat, z = z_fine)
}

phys_r <- Psi_10 %*% draws_pc + col_mean   # (m, 100)
mu <- apply(phys_r, 1, median)
sdv <- apply(phys_r, 1, sd); sdv[sdv <= 0 | !is.finite(sdv)] <- 1e-6
standardized <- (phys_r - mu) / sdv
c_q <- quantile(apply(abs(standardized), 2, max), 0.95)
lower <- mu - c_q * sdv; upper <- mu + c_q * sdv
cat(sprintf("Joint envelope multiplier c_0.95=%.3f\n", c_q))

mu_grid <- field_from_r_order(mu); lower_grid <- field_from_r_order(lower); upper_grid <- field_from_r_order(upper)
zlim <- c(-1, 1) * max(abs(range(c(lower_grid, upper_grid), na.rm = TRUE)))
zlim_mu <- c(-3, 3)
mu_grid_plot <- pmin(pmax(mu_grid, zlim_mu[1]), zlim_mu[2])

n_bins <- 24
make_bc <- function(zl) list(breaks = seq(zl[1], zl[2], length.out = n_bins + 1),
                              cols = colorRampPalette(rev(brewer.pal(11, "RdBu")))(n_bins))
bc_mu <- make_bc(zlim_mu); bc_band <- make_bc(zlim)

mu_s <- smooth_grid(mu_grid_plot); lo_s <- smooth_grid(lower_grid); hi_s <- smooth_grid(upper_grid)

pdf("chris_exact_oct2026.pdf", width = 22, height = 5.2)
par(mfrow = c(1, 3), mar = c(2, 3, 3, 1.5))
image.plot(mu_s$x, mu_s$y, mu_s$z, asp = 1, breaks = bc_mu$breaks, col = bc_mu$cols, xlab = "", ylab = "",
           main = "Chris-exact (p=10,K=5,W=36,D=4): median SSTA, Oct 2026",
           horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
image.plot(lo_s$x, lo_s$y, lo_s$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
           main = "joint 95% lower", horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
image.plot(hi_s$x, hi_s$y, hi_s$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
           main = "joint 95% upper", horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
dev.off()
cat("Saved chris_exact_oct2026.pdf and chris_exact_oct2026.rds\nDone.\n")
