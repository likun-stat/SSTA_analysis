# ============================================================================
# fno_dstm2d_fit_h6.R
#
# Final FNO-DST fit on the raw SSTA field (direct-h6, history=3), using the
# EXACT SAME chronological train/valid/test month boundaries as the GRU-EOF
# model (from eof_gru_data.RData: 473/101/102 months) so the two models are
# compared on identical data. Evaluated two ways:
#   (1) native grid: mean-forecast MSPE over ocean cells.
#   (2) projected onto the SAME K=27 EOF basis the GRU model forecasts in
#       (Psi_K, col_mean from eof_results.RData), scored with the identical
#       CRPS/twCRPS/coverage machinery used for the GRU-EOF results -- a
#       fully apples-to-apples comparison that also tests whether the EOF
#       truncation discarded forecast-relevant information.
# ============================================================================

source("fno_dstm2d_direct_h6.R")
source("eval_crps_utils.R")
load("eof_results.RData")        # Psi_K, col_mean, K_selected (EOF's own R-native ocean order)
load("eof_gru_data.RData")       # n_train, n_valid, n_test (month boundaries shared with the GRU model)
load("perm_t2r.RData")           # perm_t2r: torch-canonical ocean order -> EOF's R-native ocean order

X_trainvalid <- X_grid_flat[seq_len(n_train + n_valid), , drop = FALSE]
X_test <- X_grid_flat[seq.int(n_train + n_valid + 1L, nrow(X_grid_flat)), , drop = FALSE]
inner_train_frac <- n_train / (n_train + n_valid)
cat(sprintf("X_trainvalid=%d rows, X_test=%d rows, inner_train_frac=%.4f\n",
            nrow(X_trainvalid), nrow(X_test), inner_train_frac))

t0 <- Sys.time()
fit <- fit_fno_dstm2d_direct(
  X_trainvalid, history = 3L, horizon = 6L,
  modes_lon = 10L, modes_lat = 8L, modes_time = 2L, width = 32L, batch_size = 16L,
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

saveRDS(list(fit_history = fit$history, pred = pred, CFG = list(
  history = 3L, horizon = 6L, modes_lon = 10L, modes_lat = 8L, modes_time = 2L, width = 32L,
  jitter = COV_JITTER, rank_r = RANK_R, lengthscale_km = COV_LENGTHSCALE_KM
)), "fno_dstm2d_h6_raw.rds")
cat("Saved fno_dstm2d_h6_raw.rds\n")

# ----------------------------------------------------------------------
# Native-grid evaluation (mean-forecast MSPE over ocean cells).
# ----------------------------------------------------------------------
mspe_native <- mean((pred$pred_mean - pred$truth)^2)
cat(sprintf("\nNative-grid mean-forecast MSPE (ocean cells): %.4f\n", mspe_native))

# ----------------------------------------------------------------------
# Project every ensemble draw (and the truth) onto the K=27 EOF basis, then
# score with the SAME machinery used for the GRU-EOF-h6 results.
# ----------------------------------------------------------------------
n_origin <- dim(pred$pred_array)[1]; K <- ncol(Psi_K)
pred_array_eof <- array(NA_real_, dim = c(n_origin, K, N_ENS))
pred_mean_eof <- matrix(NA_real_, n_origin, K)

for (i in seq_len(n_origin)) {
  draws_eof_order <- pred$pred_array[i, , ][perm_t2r, ]          # (m, n_ens), reordered to EOF's ocean order
  centered <- draws_eof_order - col_mean                          # (m, n_ens)
  pred_array_eof[i, , ] <- crossprod(Psi_K, centered)             # (K, n_ens)

  mean_eof_order <- pred$pred_mean[i, ][perm_t2r]
  pred_mean_eof[i, ] <- as.numeric(crossprod(Psi_K, mean_eof_order - col_mean))
}

# Ground truth in EOF space: project the actual observed test-target field
# the same way (rather than re-reading A_K, since this keeps origin/target
# indexing exactly aligned with what predict_fno_dstm2d_direct produced).
truth_eof <- matrix(NA_real_, n_origin, K)
for (i in seq_len(n_origin)) {
  truth_eof_order <- pred$truth[i, ][perm_t2r]
  truth_eof[i, ] <- as.numeric(crossprod(Psi_K, truth_eof_order - col_mean))
}

thr <- pooled_extreme_thresholds(truth_eof)
u90 <- thr["u90"]; u95 <- thr["u95"]
scores_fno <- summarize_extreme_scores(pred_array_eof, truth_eof, pred_mean_eof, u90, u95)
cov_fno <- empirical_coverage(pred_array_eof, truth_eof)

metrics <- c("MSE_model", "CRPS", "twCRPS_u90", "twCRPS_u95", "MAE_extreme", "CRPS_extreme")
cat("\n================ FNO-DST (raw field, projected onto K=27 EOF basis) ================\n")
print(data.frame(metric = metrics, value = sapply(metrics, function(m) sprintf("%.4f", scores_fno[[m]]))), row.names = FALSE)
cat("Coverage:\n"); print(round(cov_fno, 4))

# ----------------------------------------------------------------------
# Side-by-side comparison against the GRU-EOF-h6 pooled-ensemble results.
# ----------------------------------------------------------------------
gru <- readRDS("eof_gru_h6_comparison_results.rds")
comparison <- data.frame(
  metric = metrics,
  GRU_EOF_h6_pooled = sapply(metrics, function(m) sprintf("%.4f", gru$pooled_scores[[m]])),
  FNO_DST_h6_projected = sapply(metrics, function(m) sprintf("%.4f", scores_fno[[m]]))
)
cat("\n================ GRU-EOF vs FNO-DST (both scored in EOF-coefficient space, h=6) ================\n")
print(comparison, row.names = FALSE)

saveRDS(list(
  pred_array_eof = pred_array_eof, pred_mean_eof = pred_mean_eof, truth_eof = truth_eof,
  scores_fno = scores_fno[metrics], cov_fno = cov_fno, mspe_native = mspe_native, comparison = comparison
), "fno_dstm2d_h6_eof_comparison.rds")
cat("\nSaved fno_dstm2d_h6_eof_comparison.rds\nDone.\n")
