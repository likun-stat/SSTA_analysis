# ============================================================================
# fno_dstm2d_tune_h6.R
#
# Hyperparameter grid search for FNO-DST, matching the rigor applied to both
# GRU-EOF variants (9 configs, selected by held-out validation CRPS). Grid:
# width (network capacity) x (modes_lon, modes_lat) spatial mode-capacity --
# the two architecture knobs most relevant to the "is FNO-DST overparameterized
# relative to ~470 training windows" hypothesis from the three-way comparison.
# modes_time is left at 2 (already the max allowed given history=3), and
# RANK_R / COV_LENGTHSCALE_KM are left at their existing values (40, 2500km)
# -- those govern covariance fidelity, not model capacity, and are out of
# scope for this pass.
#
# Reduced training budget during search (mean_epochs=40, nll_epochs=25,
# patience 8/15) vs. the full final-fit budget (100/50, patience 12/30) --
# same "cheap search, full final fit" pattern used for both GRU models.
#
# Selection criterion computed directly from fit$windows / fit$split$valid
# (the exact validation window indices fit_fno_dstm2d_direct already carved
# out internally for its own early stopping) -- deliberately NOT
# reconstructing a raw-time-series slice and re-deriving windows from it,
# to avoid the kind of index-arithmetic bug already caught once this session.
# ============================================================================

source("fno_dstm2d_direct_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")   # n_train, n_valid, n_test, inner_train_frac

X_trainvalid <- X_grid_flat[seq_len(n_train + n_valid), , drop = FALSE]
cat(sprintf("X_trainvalid=%d rows, inner_train_frac=%.4f\n", nrow(X_trainvalid), inner_train_frac))

N_ENS_TUNE <- 100L
SEED_TUNE <- 123L

# ----------------------------------------------------------------------
# Score a fitted model on a given subset of already-built windows (no raw-X
# reconstruction): standardize -> reshape to grid -> forward -> low-rank
# ensemble draw -> destandardize -> CRPS on ocean cells. Mirrors
# predict_fno_dstm2d_direct's body exactly, just fed pre-built windows.
# ----------------------------------------------------------------------
score_windows_crps <- function(fit, idx, n_ens = N_ENS_TUNE, seed = 999L) {
  set.seed(seed); torch_manual_seed(as.integer(seed))
  x_std <- standardize_windows(fit$windows$x[idx, , , drop = FALSE], fit$standardizer)
  x_grid <- reshape_flat_to_grid(x_std)
  y_raw_ocean <- fit$windows$y[idx, ocean_idx_t, drop = FALSE]

  n <- length(idx)
  mu_std <- matrix(NA_real_, n, m); sd_std <- matrix(NA_real_, n, m)
  fit$mean_model$eval(); fit$scale_model$eval()
  with_no_grad({
    xb <- torch_tensor(x_grid, dtype = torch_float())$to(device = fit$device)
    mu <- fit$mean_model(xb, ocean_mask_2d$to(device = fit$device))
    sd_full <- fit$scale_model(xb$permute(c(1, 4, 2, 3)))
    mu_std[, ] <- as_array(torch_flatten(mu, start_dim = 2)[, ocean_idx_t]$cpu())
    sd_std[, ] <- as_array(torch_flatten(sd_full, start_dim = 2)[, ocean_idx_t]$cpu())
  })

  Fbase_np <- as.matrix(Fbase); r <- ncol(Fbase_np)
  pred_array <- array(NA_real_, dim = c(n, m, n_ens))
  pred_mean <- matrix(NA_real_, n, m)
  for (i in seq_len(n)) {
    Z <- matrix(rnorm(r * n_ens), r, n_ens)
    eps_shared <- Fbase_np %*% Z
    eps_diag <- matrix(rnorm(m * n_ens), m, n_ens) * sqrt(fit$jitter)
    draws_std <- mu_std[i, ] + sd_std[i, ] * (eps_shared + eps_diag)
    draws_orig <- draws_std * fit$standardizer$scale[ocean_idx_t] + fit$standardizer$center[ocean_idx_t]
    pred_array[i, , ] <- draws_orig
    pred_mean[i, ] <- rowMeans(draws_orig)
  }
  crps_mat <- crps_fair_array(pred_array, y_raw_ocean)
  list(crps = mean(crps_mat, na.rm = TRUE), mspe = mean((pred_mean - y_raw_ocean)^2, na.rm = TRUE))
}

evaluate_config <- function(width, modes_lon, modes_lat, modes_time = 2L) {
  t0 <- Sys.time()
  fit <- fit_fno_dstm2d_direct(
    X_trainvalid, history = 3L, horizon = 6L,
    modes_lon = modes_lon, modes_lat = modes_lat, modes_time = modes_time, width = width, batch_size = 16L,
    learning_rate = 1e-3, weight_decay = 1e-4, lr_step_size = 10L, lr_gamma = 0.5,
    mean_epochs = 40L, nll_epochs = 25L, mean_patience = 8L, nll_patience = 15L, mean_update_every = 5L,
    train_frac = inner_train_frac, valid_frac = 1 - inner_train_frac,
    jitter = COV_JITTER, lambda_var = 10, seed = SEED_TUNE
  )
  fit_time <- as.numeric(Sys.time() - t0, units = "secs")

  sc <- score_windows_crps(fit, fit$split$valid, n_ens = N_ENS_TUNE, seed = 999L)
  cat(sprintf("[width=%d modes_lon=%d modes_lat=%d] fit %.1fs, params=%d, CRPS(valid)=%.4f, mspe=%.4f\n",
              width, modes_lon, modes_lat, fit_time, count_parameters(fit$mean_model), sc$crps, sc$mspe))
  list(width = width, modes_lon = modes_lon, modes_lat = modes_lat, fit_time = fit_time,
       n_params = count_parameters(fit$mean_model), crps = sc$crps, mspe = sc$mspe)
}

grid <- expand.grid(width = c(16L, 32L, 64L), mode_cap = c("low", "mid", "high"))
mode_lookup <- list(low = c(6L, 5L), mid = c(10L, 8L), high = c(14L, 11L))

cat(sprintf("\n================ FNO-DST tuning grid: %d configs ================\n", nrow(grid)))
results <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]; ml <- mode_lookup[[g$mode_cap]]
  cat(sprintf("\n--- config %d/%d: width=%d mode_cap=%s (modes_lon=%d, modes_lat=%d) ---\n",
              i, nrow(grid), g$width, g$mode_cap, ml[1], ml[2]))
  results[[i]] <- evaluate_config(g$width, ml[1], ml[2])
}

tab <- do.call(rbind, lapply(results, as.data.frame))
tab <- tab[order(tab$crps), ]
cat("\n================ FNO-DST tuning results, sorted by validation CRPS ================\n")
print(tab, row.names = FALSE)
winner <- tab[1, ]
cat(sprintf("\nWinner: width=%d, modes_lon=%d, modes_lat=%d (CRPS=%.4f, %d params)\n",
            winner$width, winner$modes_lon, winner$modes_lat, winner$crps, winner$n_params))

saveRDS(list(tab = tab, winner = winner), "fno_dstm2d_tune_h6_results.rds")
cat("Saved fno_dstm2d_tune_h6_results.rds\nDone.\n")
