# ============================================================================
# deesn_predict_oct2026.R
#
# Genuine October-2026 forecast from D-EESN (winning config: numLayers=7,
# layerOneDimRed=15; E=500 reservoirs, bumped up from 100 to match every
# GRU variant's own 500-member-per-seed convention -- see deesn_fit_h6.R's
# header for why 500 independent reservoirs, not 500 draws, is the faithful
# way to scale this model's ensemble), trained on ALL 676 known months
# (through April 2026) -- same convention as every other model's
# final_predict_*.R / final_plot_oct2026.R: single origin = the last known
# month (April 2026, absolute row 676), forecasting h=6 months ahead to a
# target month (row 682) that does not exist in the data.
# ============================================================================

source("deesn_eof_h6.R")
load("eof_gru_data.RData")

tune <- readRDS("deesn_tune_h6_results.rds")
CFG <- list(numLayers = tune$winner$numLayers, layerOneDimRed = tune$winner$layerOneDimRed)
cat(sprintf("Config: numLayers=%d, layerOneDimRed=%d\n", CFG$numLayers, CFG$layerOneDimRed))

p <- ncol(A_K)
TAU <- 6L; TAU_EMBED <- TAU
E_FINAL <- 500L
Tt <- nrow(A_K)
stopifnot(as.character(time_dates[Tt]) == "2026-04-01")

train_target_end <- Tt                 # use every real observed target for training
test_origins <- Tt                     # single origin: the last known month (April 2026)

des <- build_deesn_design(A_K, train_target_end, test_origins, TAU_EMBED, m = M_EMBED)
cat(sprintf("curTrainLen=%d, lenInSampleEmb=%d, curLenInbetween=%d\n", des$curTrainLen, des$lenInSampleEmb, des$curLenInbetween))

t0 <- Sys.time()
arr_coef <- fit_deesn_ensemble(des, p = p, numLayers = CFG$numLayers, layerOneDimRed = CFG$layerOneDimRed,
                                E = E_FINAL, seed0 = 1L)   # (1, p, 100)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time: %.2f min\n", fit_time))

draws_pc <- arr_coef[1, , ]   # (p, E) -- same convention as every other model's draws_pc
pc1 <- draws_pc[1, ]
cat(sprintf("\nOct-2026 EOF1 (D-EESN): median=%.1f, mean=%.1f, [5%%,95%%]=[%.1f,%.1f]\n",
            median(pc1), mean(pc1), quantile(pc1, 0.05), quantile(pc1, 0.95)))
cat("(compare: GRU-iterated=43.4, GRU-direct=36.0, tuned Dual-Stream NO=19.8, Chris-exact DSNO=35.0, historical record=54.5)\n")

saveRDS(list(draws_pc = draws_pc, CFG = CFG, p = p), "final_deesn_oct2026.rds")

# ------------------------------------------------------------------
# Physical-space reconstruction + joint-envelope 3-panel figure --
# identical method/style to final_plot_oct2026.R / chris_exact_replicate.R.
# ------------------------------------------------------------------
load("eof_results.RData")
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

phys_r <- Psi_K %*% draws_pc + col_mean   # (m, 100)
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

pdf("final_deesn_oct2026.pdf", width = 22, height = 5.2)
par(mfrow = c(1, 3), mar = c(2, 3, 3, 1.5))
image.plot(mu_s$x, mu_s$y, mu_s$z, asp = 1, breaks = bc_mu$breaks, col = bc_mu$cols, xlab = "", ylab = "",
           main = "D-EESN (numLayers=7, layerOneDimRed=15): median SSTA, Oct 2026",
           horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
image.plot(lo_s$x, lo_s$y, lo_s$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
           main = "joint 95% lower", horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
image.plot(hi_s$x, hi_s$y, hi_s$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
           main = "joint 95% upper", horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
dev.off()
cat("Saved final_deesn_oct2026.pdf and final_deesn_oct2026.rds\nDone.\n")
