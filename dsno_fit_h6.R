# ============================================================================
# dsno_fit_h6.R
#
# Final Dual-Stream Neural Operator test-set evaluation, using the tuning
# winner (K, D; W=36 fixed) and E=500 ensemble members -- bumped up from the
# paper's own example ensemble size (Sec 6: "Neural Operator Ensemble (100
# Members)") to match every GRU variant's own 500-member-per-seed convention
# elsewhere in Table 1. Note this isn't a like-for-like "more seeds" in the
# GRU sense: Dual-Stream NO has no per-network stochastic output (no
# innovation term), so each of its E members is one deterministic point
# forecast -- E plays the role of a GRU's SEED COUNT, not its within-seed
# Monte Carlo draw count (there is nothing stochastic to draw from within
# one trained network). E=500 independent networks is the faithful way to
# scale this up, not "500 draws from 100 networks."
#
# Trained on the SAME 574-month train+valid span as every other model.
# Scored on the SAME 94 target months as every other model's test set
# (absolute rows 583:676 of the full 676-month series -- the exact target
# set direct_hstep_ensemble_gru()/predict_*() use elsewhere in this study),
# via build_features_for_origins(), rather than this model's own natural
# (and differently-sized) window -- for an apples-to-apples comparison.
#
# Truth correction: scores against the TRUE physical field (rebuilt from
# the source NetCDF), not a reconstruction through this model's own p=27
# EOF basis -- matching deesn_fit_h6.R's convention. An earlier version of
# this script scored against truth = Psi_K %*% (EOF-space target), which
# let the model's own truncation define its own "truth" and never
# penalized spatial variance outside that truncation; this inflated
# DSNO's apparent accuracy, especially on extreme-value metrics.
# ============================================================================

source("dsno_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")

p <- ncol(A_K)
X_trainvalid_A <- A_K[seq_len(n_train + n_valid), , drop = FALSE]  # 574 months
TAU <- 6L
E_FINAL <- 500L
W_FIXED <- 36L

tune <- readRDS("dsno_tune_h6_results.rds")
CFG <- list(K = tune$winner$K, D = tune$winner$D, W = W_FIXED)
cat(sprintf("Winning config: K=%d, D=%d, W=%d\n", CFG$K, CFG$D, CFG$W))

B <- build_legendre_basis(CFG$W, CFG$D)
ds <- build_dual_stream(X_trainvalid_A, CFG$K, CFG$W, CFG$D, TAU, B)
d_in <- p * (CFG$K + 1 + CFG$D)
cat(sprintf("Training windows: %d, d_in=%d\n", nrow(ds$X), d_in))

t0 <- Sys.time()
fits <- fit_dsno_ensemble(ds$X, ds$Y, E = E_FINAL, d_in = d_in, p = p, seed0 = 1L,
                           epochs = 500L, patience = 20L, verbose_every = 501)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time: %.1f min\n", fit_time))

# ---- Test-set targets: absolute rows 583:676 (matching every other model) ----
target_rows <- 583:676
test_origins <- target_rows - TAU
test_feat <- build_features_for_origins(A_K, test_origins, CFG$K, CFG$W, CFG$D, B, TAU)

t1 <- Sys.time()
arr_coef <- predict_dsno_ensemble_array(fits, test_feat$X)  # (94, p, E)
n_origin <- length(test_origins)
pred_array <- array(NA_real_, dim = c(n_origin, nrow(Psi_K), E_FINAL))
for (e in seq_len(E_FINAL)) pred_array[, , e] <- t(Psi_K %*% t(arr_coef[, , e]) + col_mean)
pred_mean <- apply(pred_array, c(1, 2), mean)
pred_time <- as.numeric(Sys.time() - t1, units = "secs")
cat(sprintf("Prediction time: %.1fs\n", pred_time))

# True physical-space truth, rebuilt from the source NetCDF (same pattern
# as deesn_fit_h6.R) -- NOT Psi_K %*% test_feat$Y, which would be this
# model's own EOF-truncated echo of the truth rather than the truth itself.
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
cat("\n================ Dual-Stream NO, PHYSICAL grid space (h=6, test set) ================\n")
print(data.frame(metric = metrics, value = sapply(metrics, function(mm) sprintf("%.4f", scores[[mm]]))), row.names = FALSE)
cat("Coverage:\n"); print(round(cov, 4))

saveRDS(list(scores = scores[metrics], cov = cov, CFG = CFG, fit_time_min = fit_time, pred_time_sec = pred_time,
             tune_time_sec = sum(tune$tab$fit_time)),
        "dsno_h6_comparison.rds")
cat("\nSaved dsno_h6_comparison.rds\nDone.\n")
