# ============================================================================
# final_predict_fno_dst.R
#
# Final FNO-DST forecast for October 2026, ALL 676 months, tuned winner
# (width=16, modes_lon=10, modes_lat=8). 10 seeds pooled, n_ens=1000 each.
# Same padding trick: X_test = [Feb,Mar,Apr 2026 grid rows; 6 dummy rows].
# Forecast cost collapses to ~1/94th of the earlier 94-origin runs since
# there is now only one origin to predict.
# ============================================================================

source("fno_dstm2d_direct_h6.R")
tune <- readRDS("fno_dstm2d_tune_h6_results.rds")
CFG <- list(width = tune$winner$width, modes_lon = tune$winner$modes_lon, modes_lat = tune$winner$modes_lat)
cat(sprintf("Using tuned config: width=%d, modes_lon=%d, modes_lat=%d\n", CFG$width, CFG$modes_lon, CFG$modes_lat))

N_SEEDS <- 10L
N_ENS <- 1000L

X_all <- X_grid_flat
last3 <- X_all[(nrow(X_all) - 2):nrow(X_all), , drop = FALSE]
dummy <- matrix(rep(colMeans(X_all), 6), nrow = 6, byrow = TRUE)
X_test_padded <- rbind(last3, dummy)

run_one <- function(seed) {
  t0 <- Sys.time()
  fit <- fit_fno_dstm2d_direct(
    X_all, history = 3L, horizon = 6L, modes_lon = CFG$modes_lon, modes_lat = CFG$modes_lat,
    modes_time = 2L, width = CFG$width, batch_size = 16L, learning_rate = 1e-3, weight_decay = 1e-4,
    lr_step_size = 10L, lr_gamma = 0.5, mean_epochs = 100L, nll_epochs = 50L, mean_patience = 12L,
    nll_patience = 30L, mean_update_every = 5L, train_frac = 0.85, valid_frac = 0.15,
    jitter = COV_JITTER, lambda_var = 10, seed = seed
  )
  fit_time <- as.numeric(Sys.time() - t0, units = "mins")
  pred <- predict_fno_dstm2d_direct(fit, X_test_padded, n_ens = N_ENS, seed = seed + 1000)
  cat(sprintf("[fno seed=%d] fit %.1fmin\n", seed, fit_time))
  pred$pred_array[1, , ]  # (m, N_ENS), torch-canonical ocean order
}

cat("Training FNO-DST on ALL 676 months, forecasting Oct 2026 (10 seeds)...\n")
draws_list <- vector("list", N_SEEDS)
for (s in seq_len(N_SEEDS)) { cat(sprintf("--- seed %d/%d ---\n", s, N_SEEDS)); draws_list[[s]] <- run_one(s) }
draws_pooled <- do.call(cbind, draws_list)  # (m, N_SEEDS*N_ENS)
cat(sprintf("Pooled draws: %d x %d\n", nrow(draws_pooled), ncol(draws_pooled)))

saveRDS(list(draws_ocean_torch_order = draws_pooled, CFG = CFG, target_date = "2026-10-01"), "final_fno_dst_oct2026.rds")
cat("Saved final_fno_dst_oct2026.rds\nDone.\n")
