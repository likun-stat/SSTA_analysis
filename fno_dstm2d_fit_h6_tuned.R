# ============================================================================
# fno_dstm2d_fit_h6_tuned.R
#
# Final FNO-DST fit using the grid-search winner (width=16, modes_lon=10,
# modes_lat=8 -- fno_dstm2d_tune_h6_results.rds), full training budget
# (mean_epochs=100, nll_epochs=50, patience 12/30 -- same as the original
# untuned fit), evaluated in both EOF-coefficient space and physical grid
# space.
#
# Trimmed from the original project's version: dropped the three-way
# comparison against two GRU-model result files (physical_space_comparison_h6
# _results.rds, gru_eof_direct_h6_results.rds), which are GRU-derived outputs
# not included in this directory (GRU is intentionally excluded here). FNO-DST
# 's own metrics, printed below, are unaffected.
# ============================================================================

source("fno_dstm2d_direct_h6.R")
source("eval_crps_utils.R")
load("eof_results.RData")
load("eof_gru_data.RData")
load("perm_t2r.RData")
tune <- readRDS("fno_dstm2d_tune_h6_results.rds")
CFG <- list(width = tune$winner$width, modes_lon = tune$winner$modes_lon, modes_lat = tune$winner$modes_lat)
cat(sprintf("Tuned config: width=%d, modes_lon=%d, modes_lat=%d\n", CFG$width, CFG$modes_lon, CFG$modes_lat))

X_trainvalid <- X_grid_flat[seq_len(n_train + n_valid), , drop = FALSE]
X_test <- X_grid_flat[seq.int(n_train + n_valid + 1L, nrow(X_grid_flat)), , drop = FALSE]

t0 <- Sys.time()
fit <- fit_fno_dstm2d_direct(
  X_trainvalid, history = 3L, horizon = 6L,
  modes_lon = CFG$modes_lon, modes_lat = CFG$modes_lat, modes_time = 2L, width = CFG$width, batch_size = 16L,
  learning_rate = 1e-3, weight_decay = 1e-4, lr_step_size = 10L, lr_gamma = 0.5,
  mean_epochs = 100L, nll_epochs = 50L, mean_patience = 12L, nll_patience = 30L, mean_update_every = 5L,
  train_frac = inner_train_frac, valid_frac = 1 - inner_train_frac,
  jitter = COV_JITTER, lambda_var = 10, seed = 123L
)
cat(sprintf("\nTotal fit time: %.1f min\n", as.numeric(Sys.time() - t0, units = "mins")))

N_ENS <- 500L
pred <- predict_fno_dstm2d_direct(fit, X_test, n_ens = N_ENS, seed = 999L)
cat(sprintf("Predicted %d test origins, %d ocean cells, %d ensemble members\n",
            dim(pred$pred_array)[1], dim(pred$pred_array)[2], dim(pred$pred_array)[3]))
saveRDS(list(fit_history = fit$history, pred = pred, CFG = CFG), "fno_dstm2d_h6_tuned_raw.rds")
cat("Saved fno_dstm2d_h6_tuned_raw.rds\n")

# ---- Physical-space evaluation (the trustworthy comparison) ----
n_origin <- dim(pred$pred_array)[1]; m <- dim(pred$pred_array)[2]
pred_array_r <- array(NA_real_, dim = dim(pred$pred_array))
for (i in seq_len(n_origin)) pred_array_r[i, , ] <- pred$pred_array[i, , ][perm_t2r, ]
truth_r <- pred$truth[, perm_t2r]
pred_mean_r <- pred$pred_mean[, perm_t2r]

thr <- pooled_extreme_thresholds(truth_r); u90 <- thr["u90"]; u95 <- thr["u95"]
scores_phys <- summarize_extreme_scores(pred_array_r, truth_r, pred_mean_r, u90, u95)
cov_phys <- empirical_coverage(pred_array_r, truth_r)

metrics <- c("MSE_model", "CRPS", "twCRPS_u90", "twCRPS_u95", "MAE_extreme", "CRPS_extreme")
cat("\n================ FNO-DST (TUNED), PHYSICAL grid space ================\n")
print(data.frame(metric = metrics, value = sapply(metrics, function(mm) sprintf("%.4f", scores_phys[[mm]]))), row.names = FALSE)
cat("Coverage:\n"); print(round(cov_phys, 4))

saveRDS(list(scores_phys = scores_phys[metrics], cov_phys = cov_phys, CFG = CFG),
        "fno_dstm2d_h6_tuned_comparison.rds")
cat("\nSaved fno_dstm2d_h6_tuned_comparison.rds\nDone.\n")
