# ============================================================================
# deesn_fit_h6.R
#
# Final D-EESN test-set evaluation, winning config from deesn_tune_h6.R
# (numLayers=7, layerOneDimRed=15), E=500 reservoirs -- bumped up from
# McDermott & Wikle's own Lorenz-96 ensembleLen=100 example to match every
# GRU variant's own 500-member-per-seed convention elsewhere in Table 1
# (same rationale as dsno_fit_h6.R's own E=100->500 bump: D-EESN has no
# per-reservoir stochastic output either, so E plays the role of a GRU's
# seed count, and 500 independent reservoirs is the faithful way to scale
# it up). Trained on the full 676-month series (the
# authors' own architecture needs to "see" the actual test-origin rows and
# their history as embedding INPUTS to forecast from them -- see
# deesn_eof_h6.R's build_deesn_design() docstring -- but training TARGETS
# are capped at row n_train+n_valid=574, the SAME boundary every other
# model's final fit respects (e.g. GRU-direct's own last training window
# target is exactly row 574), not merely "before the first scored test
# target" (583). An earlier version of this script capped training targets
# at row 582 instead (one before the first SCORED target, rather than one
# after the last TRAINABLE row) -- which let D-EESN's ridge regression fit
# directly on 7 extra months (575-581) that no other model's parameters
# ever see, a genuine comparability gap now closed), scored on the SAME
# 94 held-out target months (absolute rows
# 583:676) as every other model, reconstructed to TRUE physical grid space
# (from the source NetCDF, not the EOF-restricted proxy used during tuning)
# -- identical methodology to chris_exact_fit_h6.R / gru_eof_direct_h6_fit.R.
# ============================================================================

source("deesn_eof_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")   # n_train, n_valid, time_dates

tune <- readRDS("deesn_tune_h6_results.rds")
CFG <- list(numLayers = tune$winner$numLayers, layerOneDimRed = tune$winner$layerOneDimRed)
cat(sprintf("Winning config: numLayers=%d, layerOneDimRed=%d\n", CFG$numLayers, CFG$layerOneDimRed))

p <- ncol(A_K)
TAU <- 6L; TAU_EMBED <- TAU
E_FINAL <- 500L

target_rows <- 583:676                 # same 94 test-set target months as every other model
test_origins <- target_rows - TAU      # 577:670
train_target_end <- n_train + n_valid  # 574 -- SAME no-leakage boundary every other model's final fit uses

t0 <- Sys.time()
des <- build_deesn_design(A_K, train_target_end, test_origins, TAU_EMBED, m = M_EMBED)
cat(sprintf("curTrainLen=%d, lenInSampleEmb=%d, curLenInbetween=%d\n", des$curTrainLen, des$lenInSampleEmb, des$curLenInbetween))

arr_coef <- fit_deesn_ensemble(des, p = p, numLayers = CFG$numLayers, layerOneDimRed = CFG$layerOneDimRed,
                                E = E_FINAL, seed0 = 1L)   # (94, p, 100)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time (E=%d): %.2f min\n", E_FINAL, fit_time))

# ----------------------------------------------------------------------
# True physical-space truth, rebuilt from the source NetCDF exactly as in
# chris_exact_fit_h6.R / gru_eof_direct_h6_fit.R / physical_space_comparison_h6.R
# ----------------------------------------------------------------------
load("eof_results.RData")   # Psi_K, col_mean, ocean_idx
library(ncdf4)
nc <- nc_open("sst_anomalies_extracted.nc"); sst <- ncvar_get(nc, "sst"); nc_close(nc)
Tt <- dim(sst)[3]
sst_flat_r <- matrix(NA_real_, nrow = Tt, ncol = dim(sst)[1] * dim(sst)[2])
for (tt in seq_len(Tt)) sst_flat_r[tt, ] <- as.vector(sst[, , tt])
X_ocean_r <- sst_flat_r[, ocean_idx, drop = FALSE]
truth_phys <- X_ocean_r[target_rows, , drop = FALSE]   # (94, m)
stopifnot(nrow(truth_phys) == length(test_origins))

m <- nrow(Psi_K)
n_origin <- length(test_origins)
pred_array <- array(NA_real_, dim = c(n_origin, m, E_FINAL))
for (e in seq_len(E_FINAL)) pred_array[, , e] <- t(Psi_K %*% t(arr_coef[, , e]) + col_mean)
pred_mean <- apply(pred_array, c(1, 2), mean)

thr <- pooled_extreme_thresholds(truth_phys); u90 <- thr["u90"]; u95 <- thr["u95"]
scores <- summarize_extreme_scores(pred_array, truth_phys, pred_mean, u90, u95)
cov <- empirical_coverage(pred_array, truth_phys)

metrics <- c("MSE_model", "CRPS", "twCRPS_u90", "twCRPS_u95", "MAE_extreme", "CRPS_extreme")
cat("\n================ D-EESN, PHYSICAL grid space (h=6, test set) ================\n")
print(data.frame(metric = metrics, value = sapply(metrics, function(mm) sprintf("%.4f", scores[[mm]]))), row.names = FALSE)
cat("Coverage:\n"); print(round(cov, 4))

saveRDS(list(scores = scores[metrics], cov = cov, CFG = CFG, fit_time_min = fit_time, E_FINAL = E_FINAL),
        "deesn_h6_comparison.rds")
cat("\nSaved deesn_h6_comparison.rds\nDone.\n")
