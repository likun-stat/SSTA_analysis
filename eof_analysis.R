# ============================================================================
# eof_analysis.R
#
# EOF (spatial-PCA) analysis of the SSTA field (sst_anomalies_extracted.nc,
# Jan 1970 - Apr 2026), reducing dimension over SPACE (locations), not time.
# Follows the setup in the project write-up:
#
#   C_hat^(0) = (1/(T-1)) sum_j Z_tj Z_tj^T = Psi Lambda Psi^T
#
# with EOFs = columns of Psi (spatial patterns), eigenvalues Lambda (variance
# along each EOF), and PC series a_k(t_j) = psi_k^T Z_tj.
#
# Number of retained modes K is chosen by a permutation scree test (Cressie
# & Wikle 2011): independently permute the time order of EACH location's own
# series (destroys space-time covariance, preserves each location's own
# marginal distribution), recompute the eigenvalue spectrum many times, and
# retain the leading modes whose observed %variance-explained exceeds the
# MEDIAN of the permuted null spectrum at that same rank.
#
# Efficiency note: T=676 << m=2512 ocean grid cells, so eigen-decomposing
# the T x T Gram matrix G = (1/(T-1)) X X^T is far cheaper than the m x m
# spatial covariance C = (1/(T-1)) X^T X -- they share the same nonzero
# eigenvalues, and the spatial EOFs are recovered as
#   psi_k = X^T v_k / sqrt((T-1) * lambda_k)
# where v_k is G's k-th eigenvector. This trick is used both for the
# observed data and inside every permutation replicate.
# ============================================================================

library(ncdf4)
library(fields)

set.seed(20260928)

# ------------------------------------------------------------------
# 1. Load data, build the ocean-only space-time matrix X (T x m)
# ------------------------------------------------------------------
nc <- nc_open("sst_anomalies_extracted.nc")
sst <- ncvar_get(nc, "sst")        # array (lon, lat, time)
lat <- ncvar_get(nc, "lat")
lon <- ncvar_get(nc, "lon")
time_raw <- ncvar_get(nc, "time")
time_units <- ncatt_get(nc, "time", "units")$value
nc_close(nc)

nlon <- length(lon); nlat <- length(lat); Tt <- dim(sst)[3]
origin <- as.Date(sub("days since ", "", time_units))
time_dates <- origin + time_raw

# Flatten space (lon fastest, then lat -- matches R's column-major array
# order for sst[,,tt], so field_from_psi() below inverts it exactly).
sst_flat <- matrix(NA_real_, nrow = Tt, ncol = nlon * nlat)
for (tt in seq_len(Tt)) sst_flat[tt, ] <- as.vector(sst[, , tt])

ocean_idx <- which(!apply(sst_flat, 2, function(v) all(is.na(v))))
stopifnot(!anyNA(sst_flat[, ocean_idx]))  # confirmed: no partial-NA columns (checked interactively)

X_raw <- sst_flat[, ocean_idx, drop = FALSE]   # (T x m)
m <- ncol(X_raw)
cat(sprintf("T=%d, m=%d ocean grid cells (of %d total, %d land)\n",
            Tt, m, nlon * nlat, nlon * nlat - m))

# ------------------------------------------------------------------
# 2. Temporal centering. The field is already an anomaly relative to the
#    1981-2010 climatology, but that baseline need not have exactly zero
#    mean over the full 1970-2026 record (e.g. the warming trend), so we
#    explicitly remove each location's own time mean before computing the
#    spatial covariance, as specified.
# ------------------------------------------------------------------
col_mean <- colMeans(X_raw)
X <- sweep(X_raw, 2, col_mean, "-")

# ------------------------------------------------------------------
# 3. Fast eigen-decomposition via the T x T Gram matrix.
# ------------------------------------------------------------------
G_obs <- tcrossprod(X) / (Tt - 1)                 # T x T
eig_obs <- eigen(G_obs, symmetric = TRUE)
lambda_obs <- pmax(eig_obs$values, 0)
pve_obs <- lambda_obs / sum(lambda_obs) * 100

K_max_keep <- 40  # more than enough for the scree cutoff + the 8 requested for plotting
V_obs <- eig_obs$vectors[, seq_len(K_max_keep), drop = FALSE]

psi_from_v <- function(vk, lambda_k) crossprod(X, vk) / sqrt((Tt - 1) * lambda_k)
Psi <- sapply(seq_len(K_max_keep), function(k) psi_from_v(V_obs[, k], lambda_obs[k]))
A <- sapply(seq_len(K_max_keep), function(k) sqrt((Tt - 1) * lambda_obs[k]) * V_obs[, k])
colnames(Psi) <- colnames(A) <- paste0("EOF", seq_len(K_max_keep))

# ------------------------------------------------------------------
# 4. Permutation scree test.
# ------------------------------------------------------------------
B <- 1000
K_rank_check <- 30   # null distribution only needed out to a modest rank

t0 <- Sys.time()
perm_pve <- matrix(NA_real_, nrow = B, ncol = K_rank_check)
for (b in seq_len(B)) {
  Xp <- X
  for (j in seq_len(m)) Xp[, j] <- Xp[sample.int(Tt), j]
  Gp <- tcrossprod(Xp) / (Tt - 1)
  lp <- pmax(eigen(Gp, symmetric = TRUE, only.values = TRUE)$values, 0)
  perm_pve[b, ] <- (lp / sum(lp) * 100)[seq_len(K_rank_check)]
  if (b %% 100 == 0) {
    cat(sprintf("permutation %d/%d (%.1fs elapsed)\n", b, B,
                as.numeric(Sys.time() - t0, units = "secs")))
  }
}
perm_median <- apply(perm_pve, 2, median)
perm_q05 <- apply(perm_pve, 2, quantile, probs = 0.05)
perm_q95 <- apply(perm_pve, 2, quantile, probs = 0.95)

obs_check <- pve_obs[seq_len(K_rank_check)]
first_fail <- which(obs_check <= perm_median)[1]
K_selected <- if (is.na(first_fail)) K_rank_check else first_fail - 1L
cat(sprintf("\nPermutation scree test (B=%d): selected K = %d\n", B, K_selected))

scree_tab <- data.frame(
  rank = seq_len(K_rank_check), pve_obs = obs_check,
  perm_median = perm_median, perm_q05 = perm_q05, perm_q95 = perm_q95,
  retained = seq_len(K_rank_check) <= K_selected
)
print(round(scree_tab, 3))

# ------------------------------------------------------------------
# 5. Save the first K EOF basis + PC scores, with enough metadata to map
#    a spatial vector back onto the (lon, lat) grid later.
# ------------------------------------------------------------------
K <- K_selected
Psi_K <- Psi[, seq_len(K), drop = FALSE]
A_K <- A[, seq_len(K), drop = FALSE]

save(Psi_K, A_K, lambda_obs, pve_obs, scree_tab, K_selected, B,
     lon, lat, ocean_idx, nlon, nlat, time_dates, col_mean,
     file = "eof_results.RData")
cat("Saved eof_results.RData (K =", K, "retained modes)\n")

# ------------------------------------------------------------------
# 6. Visualization: permutation scree plot; first 8 PC time series;
#    first 8 EOF spatial maps.
# ------------------------------------------------------------------
pdf("eof_scree_permutation.pdf", width = 8, height = 6)
plot(scree_tab$rank, scree_tab$pve_obs, type = "b", pch = 16, col = "black",
     xlab = "EOF rank", ylab = "% variance explained",
     main = sprintf("Permutation scree test (B=%d) -- selected K=%d", B, K_selected),
     ylim = c(0, max(scree_tab$pve_obs, scree_tab$perm_q95)))
lines(scree_tab$rank, scree_tab$perm_median, col = "red", lwd = 2)
lines(scree_tab$rank, scree_tab$perm_q05, col = "red", lty = 3)
lines(scree_tab$rank, scree_tab$perm_q95, col = "red", lty = 3)
abline(v = K_selected + 0.5, col = "gray50", lty = 2)
legend("topright", legend = c("observed", "permuted median", "permuted 5-95%"),
       col = c("black", "red", "red"), lty = c(1, 1, 3), pch = c(16, NA, NA))
dev.off()

n_show <- min(8, K)

pdf("eof_pc_timeseries.pdf", width = 10, height = 10)
par(mfrow = c(4, 2), mar = c(3, 3, 2, 1))
for (k in seq_len(n_show)) {
  plot(time_dates, A_K[, k], type = "l", col = "steelblue",
       xlab = "", ylab = "", main = sprintf("PC%d (%.1f%% var)", k, pve_obs[k]))
  abline(h = 0, col = "gray60", lty = 2)
}
dev.off()

field_from_psi <- function(psi_vec) {
  full <- rep(NA_real_, nlon * nlat)
  full[ocean_idx] <- psi_vec
  matrix(full, nrow = nlon, ncol = nlat)   # (lon, lat)
}

pdf("eof_spatial_basis.pdf", width = 12, height = 10)
par(mfrow = c(4, 2), mar = c(3, 3, 2, 5))
for (k in seq_len(n_show)) {
  Zmat <- field_from_psi(Psi_K[, k])
  zmax <- max(abs(Zmat), na.rm = TRUE)
  image.plot(lon, lat, Zmat, col = fields::tim.colors(64),
             zlim = c(-zmax, zmax), xlab = "", ylab = "",
             main = sprintf("EOF%d (%.1f%% var)", k, pve_obs[k]))
}
dev.off()

cat("Saved eof_scree_permutation.pdf, eof_pc_timeseries.pdf, eof_spatial_basis.pdf\n")
cat("Done.\n")
