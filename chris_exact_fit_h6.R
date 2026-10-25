# ============================================================================
# chris_exact_fit_h6.R
#
# Test-set evaluation of the Dual-Stream NO using C.K. Wikle's own example
# hyperparameters exactly (p=10 EOFs, K=5, W=36, D=4, E=500), mirroring
# dsno_fit_h6.R's methodology exactly (same 574-month train+valid span, same
# 94 held-out test-set target months [absolute rows 583:676], same physical-
# space CRPS/MSE/coverage machinery, same bumped-up E=500 ensemble size --
# see dsno_fit_h6.R's header for why 500 independent networks, not 500
# draws, is the faithful way to scale this model's ensemble) so the result
# is a like-for-like column for Table 1 -- NOT the tuned K=3/D=2/p=27
# configuration used elsewhere.
#
# Truth correction: as in dsno_fit_h6.R, scores against the TRUE physical
# field (from the source NetCDF), not Psi_10 %*% (this model's own
# 10-EOF-truncated target) -- the original version of this script used the
# latter, which never penalized spatial variance outside this config's own
# (even smaller than the tuned model's) truncation.
# ============================================================================

source("dsno_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")

P_CHRIS <- 10L
A_10 <- A_K[, seq_len(P_CHRIS), drop = FALSE]
Psi_10 <- Psi_K[, seq_len(P_CHRIS), drop = FALSE]

CFG <- list(p = P_CHRIS, K = 5L, W = 36L, D = 4L)
TAU <- 6L
E_FINAL <- 500L
cat(sprintf("Chris-exact config: p=%d, K=%d, W=%d, D=%d\n", CFG$p, CFG$K, CFG$W, CFG$D))

X_trainvalid_A <- A_10[seq_len(n_train + n_valid), , drop = FALSE]  # 574 months
B <- build_legendre_basis(CFG$W, CFG$D)
ds <- build_dual_stream(X_trainvalid_A, CFG$K, CFG$W, CFG$D, TAU, B)
d_in <- CFG$p * (CFG$K + 1 + CFG$D)
cat(sprintf("Training windows: %d, d_in=%d\n", nrow(ds$X), d_in))

t0 <- Sys.time()
fits <- fit_dsno_ensemble(ds$X, ds$Y, E = E_FINAL, d_in = d_in, p = CFG$p, seed0 = 1L,
                           epochs = 500L, patience = 20L, verbose_every = 501)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time: %.1f min\n", fit_time))

# ---- Test-set targets: absolute rows 583:676 (matching every other model) ----
target_rows <- 583:676
test_origins <- target_rows - TAU
test_feat <- build_features_for_origins(A_10, test_origins, CFG$K, CFG$W, CFG$D, B, TAU)

arr_coef <- predict_dsno_ensemble_array(fits, test_feat$X)  # (94, p=10, E)
n_origin <- length(test_origins)
pred_array <- array(NA_real_, dim = c(n_origin, nrow(Psi_10), E_FINAL))
for (e in seq_len(E_FINAL)) pred_array[, , e] <- t(Psi_10 %*% t(arr_coef[, , e]) + col_mean)
pred_mean <- apply(pred_array, c(1, 2), mean)

# True physical-space truth, rebuilt from the source NetCDF (same pattern
# as dsno_fit_h6.R / deesn_fit_h6.R).
library(ncdf4)
nc <- nc_open("sst_anomalies_extracted.nc"); sst <- ncvar_get(nc, "sst"); nc_close(nc)
Tt_nc <- dim(sst)[3]
sst_flat_r <- matrix(NA_real_, nrow = Tt_nc, ncol = dim(sst)[1] * dim(sst)[2])
for (tt in seq_len(Tt_nc)) sst_flat_r[tt, ] <- as.vector(sst[, , tt])
X_ocean_r <- sst_flat_r[, ocean_idx, drop = FALSE]
truth <- X_ocean_r[target_rows, , drop = FALSE]
stopifnot(nrow(truth) == length(test_origins))

thr <- pooled_extreme_thresholds(truth); u90 <- thr["u90"]; u95 <- thr["u95"]
scores <- summarize_extreme_scores(pred_array, truth, pred_mean, u90, u95)
cov <- empirical_coverage(pred_array, truth)

metrics <- c("MSE_model", "CRPS", "twCRPS_u90", "twCRPS_u95", "MAE_extreme", "CRPS_extreme")
cat("\n================ Dual-Stream NO (Chris-exact), PHYSICAL grid space (h=6, test set) ================\n")
print(data.frame(metric = metrics, value = sapply(metrics, function(mm) sprintf("%.4f", scores[[mm]]))), row.names = FALSE)
cat("Coverage:\n"); print(round(cov, 4))

saveRDS(list(scores = scores[metrics], cov = cov, CFG = CFG, fit_time_min = fit_time),
        "chris_exact_h6_comparison.rds")
cat("\nSaved chris_exact_h6_comparison.rds\nDone.\n")
