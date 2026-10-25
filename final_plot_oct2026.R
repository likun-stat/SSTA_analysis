# ============================================================================
# final_plot_oct2026.R
#
# Reconstructs each method's pooled Oct-2026 ensemble to the physical
# (lon,lat) grid, builds a JOINT (simultaneous) 95% spatial band via the
# max-statistic/envelope method, and saves a 3-panel (mean, lower, upper)
# figure per method.
#
# Joint-band method: for each ensemble member e, compute the max
# standardized deviation across ocean cells, max_j |Y_e(j)-mean(j)|/sd(j);
# take the 95th percentile of that max-statistic ACROSS the ensemble ->
# one scalar c_0.95. Band = mean +/- c_0.95 * sd (per cell). This is a
# standard simultaneous-confidence-band construction (the envelope/max-
# statistic approach from functional data analysis): by construction,
# ~95% of ensemble members lie ENTIRELY within the band at EVERY cell
# simultaneously -- a genuinely joint (not merely pointwise/marginal) 95%
# region, estimated directly and nonparametrically from the ensemble itself.
#
# Both GRU methods reconstruct via physical = col_mean + Psi_K %*% draws_pc
# (EOF/R-native ocean order, matching Psi_K's own row order). FNO-DST/
# CNN-IDE draws are in torch-canonical ocean order and are reordered via
# the validated perm_t2r before using the same field_from_psi-style grid
# reconstruction -- giving ALL FOUR methods a common (lon,lat) grid
# representation for plotting.
# ============================================================================

load("eof_results.RData")   # Psi_K, col_mean, ocean_idx, nlon, nlat, lon, lat
load("perm_t2r.RData")      # torch-canonical ocean order -> EOF/R-native ocean order

library(fields)
library(RColorBrewer)

field_from_r_order <- function(values_r_order) {
  full <- rep(NA_real_, nlon * nlat)
  full[ocean_idx] <- values_r_order
  matrix(full, nrow = nlon, ncol = nlat)
}

# Bilinearly upsamples a (nlon x nlat) grid (NA = land) onto a much finer
# grid for smooth, contourf-like display -- image()/image.plot() otherwise
# draws one flat, sharp-edged rectangle per raw 2-deg cell (a "staircase"
# look), unlike filled-contour-style plotting which interpolates between
# grid points and draws curved color-threshold boundaries. This does not
# add information the 2-deg data doesn't have; it only smooths how that
# same data is *displayed*, matching the reference figure's rendering style.
# Land cells are filled with the ocean mean before interpolating (avoiding
# NA bleeding into near-coast ocean values), then the mask itself is
# bilinearly upsampled and thresholded at 0.5 to re-mask land afterward.
smooth_grid <- function(mat, factor = 6) {
  land <- is.na(mat)
  filled <- mat; filled[land] <- mean(mat, na.rm = TRUE)
  fine_lon <- seq(min(lon), max(lon), length.out = length(lon) * factor)
  fine_lat <- seq(min(lat), max(lat), length.out = length(lat) * factor)
  z_fine <- interp.surface.grid(list(x = lon, y = lat, z = filled), list(x = fine_lon, y = fine_lat))$z
  mask_fine <- interp.surface.grid(list(x = lon, y = lat, z = matrix(as.numeric(!land), nlon, nlat)),
                                    list(x = fine_lon, y = fine_lat))$z
  z_fine[mask_fine < 0.5] <- NA
  list(x = fine_lon, y = fine_lat, z = z_fine)
}

joint_envelope_and_plot <- function(draws_r_order, method_label, out_pdf, prob = 0.95) {
  # draws_r_order: (m, n_ens) matrix, EOF/R-native ocean-cell order.
  # Center = per-cell MEDIAN (the "best estimate"), not the mean -- more
  # robust to the occasional extreme Monte Carlo draw, per request. The
  # envelope is built around this same median (median +/- c_q * sd), so the
  # reported band is self-consistent with the reported center.
  m <- nrow(draws_r_order)
  mu <- apply(draws_r_order, 1, median)
  sdv <- apply(draws_r_order, 1, sd); sdv[sdv <= 0 | !is.finite(sdv)] <- 1e-6

  standardized <- (draws_r_order - mu) / sdv           # (m, n_ens)
  max_stat <- apply(abs(standardized), 2, max)          # (n_ens,) -- per-member max over cells
  c_q <- quantile(max_stat, probs = prob)
  cat(sprintf("[%s] joint envelope multiplier c_%.2f = %.3f (vs %.3f for a pointwise z=1.96 band)\n",
              method_label, prob, c_q, qnorm(0.5 + prob / 2)))

  lower <- mu - c_q * sdv
  upper <- mu + c_q * sdv

  mu_grid <- field_from_r_order(mu)
  lower_grid <- field_from_r_order(lower)
  upper_grid <- field_from_r_order(upper)

  zlim <- range(c(lower_grid, upper_grid), na.rm = TRUE)
  zlim <- c(-1, 1) * max(abs(zlim))
  zlim_mu <- c(-3, 3)  # median is much smaller in magnitude than the joint band; its own
                       # tighter fixed scale keeps it from washing out near-white under the
                       # band's wide range.

  # Discrete (banded) diverging color scale -- 24 bins, red=warm/positive,
  # blue=cool/negative (oceanographic SSTA convention). The median panel
  # uses its own scale; the lower/upper panels share one (same physical
  # quantity, same legend).
  n_bins <- 24
  make_breaks_cols <- function(zl) list(breaks = seq(zl[1], zl[2], length.out = n_bins + 1),
                                          cols = colorRampPalette(rev(brewer.pal(11, "RdBu")))(n_bins))
  bc_mu <- make_breaks_cols(zlim_mu)
  bc_band <- make_breaks_cols(zlim)

  # asp=1 (real lon/lat degree units, equirectangular) -- the domain is
  # ~2.6:1 wide, so size the device to roughly match that per-panel shape
  # and avoid wasted letterboxing whitespace.
  # Clamp to the plotted range rather than leaving out-of-range cells blank
  # (image() would otherwise not color values outside `breaks`, which reads
  # as missing/land): values beyond the scale saturate to the endpoint
  # color, matching the reference image's arrow-capped colorbar convention.
  mu_grid_plot <- pmin(pmax(mu_grid, zlim_mu[1]), zlim_mu[2])

  # Smooth (bilinearly upsampled) versions for display -- see smooth_grid()
  # header comment; this is purely a rendering choice, not new information.
  mu_smooth <- smooth_grid(mu_grid_plot)
  lower_smooth <- smooth_grid(lower_grid)
  upper_smooth <- smooth_grid(upper_grid)

  pdf(out_pdf, width = 22, height = 5.2)
  par(mfrow = c(1, 3), mar = c(2, 3, 3, 1.5))
  image.plot(mu_smooth$x, mu_smooth$y, mu_smooth$z, asp = 1, breaks = bc_mu$breaks, col = bc_mu$cols, xlab = "", ylab = "",
             main = sprintf("%s: median SSTA, Oct 2026", method_label),
             horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
  image.plot(lower_smooth$x, lower_smooth$y, lower_smooth$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
             main = sprintf("%s: joint 95%% lower", method_label),
             horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
  image.plot(upper_smooth$x, upper_smooth$y, upper_smooth$z, asp = 1, breaks = bc_band$breaks, col = bc_band$cols, xlab = "", ylab = "",
             main = sprintf("%s: joint 95%% upper", method_label),
             horizontal = TRUE, legend.shrink = 0.9, legend.mar = 4.5, useRaster = TRUE)
  dev.off()
  cat(sprintf("Saved %s\n", out_pdf))

  list(mu = mu, lower = lower, upper = upper, c_q = c_q, mu_grid = mu_grid, lower_grid = lower_grid, upper_grid = upper_grid)
}

# ------------------------------------------------------------------
# GRU-EOF-iterated
# ------------------------------------------------------------------
if (file.exists("final_gru_iterated_oct2026.rds")) {
  d <- readRDS("final_gru_iterated_oct2026.rds")
  phys_r <- Psi_K %*% d$draws_pc + col_mean   # (m, n_ens), R-native order
  res_gi <- joint_envelope_and_plot(phys_r, "GRU-EOF-iterated", "final_gru_iterated_oct2026.pdf")
  saveRDS(res_gi, "final_gru_iterated_oct2026_envelope.rds")
}

# ------------------------------------------------------------------
# GRU-EOF-direct
# ------------------------------------------------------------------
if (file.exists("final_gru_direct_oct2026.rds")) {
  d <- readRDS("final_gru_direct_oct2026.rds")
  phys_r <- Psi_K %*% d$draws_pc + col_mean
  res_gd <- joint_envelope_and_plot(phys_r, "GRU-EOF-direct", "final_gru_direct_oct2026.pdf")
  saveRDS(res_gd, "final_gru_direct_oct2026_envelope.rds")
}

# ------------------------------------------------------------------
# FNO-DST
# ------------------------------------------------------------------
if (file.exists("final_fno_dst_oct2026.rds")) {
  d <- readRDS("final_fno_dst_oct2026.rds")
  phys_r <- d$draws_ocean_torch_order[perm_t2r, ]
  res_fno <- joint_envelope_and_plot(phys_r, "FNO-DST", "final_fno_dst_oct2026.pdf")
  saveRDS(res_fno, "final_fno_dst_oct2026_envelope.rds")
}

# ------------------------------------------------------------------
# CNN-IDE
# ------------------------------------------------------------------
if (file.exists("final_cnn_ide_oct2026.rds")) {
  d <- readRDS("final_cnn_ide_oct2026.rds")
  phys_r <- d$draws_ocean_torch_order[perm_t2r, ]
  res_cnn <- joint_envelope_and_plot(phys_r, "CNN-IDE", "final_cnn_ide_oct2026.pdf")
  saveRDS(res_cnn, "final_cnn_ide_oct2026_envelope.rds")
}

# ------------------------------------------------------------------
# Dual-Stream Neural Operator
# ------------------------------------------------------------------
if (file.exists("final_dsno_oct2026.rds")) {
  d <- readRDS("final_dsno_oct2026.rds")
  phys_r <- Psi_K %*% d$draws_pc + col_mean
  res_dsno <- joint_envelope_and_plot(phys_r, "Dual-Stream NO", "final_dsno_oct2026.pdf")
  saveRDS(res_dsno, "final_dsno_oct2026_envelope.rds")
}

cat("Done.\n")
