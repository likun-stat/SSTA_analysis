# ============================================================================
# dsno_tune_h6.R
#
# Hyperparameter grid search for the Dual-Stream Neural Operator, matching
# the same 9-config, validation-CRPS-selected pattern used for every other
# model. Grid: short-lag order K in {3,5,8} x Legendre degree count D in
# {2,4,6} -- the paper's own headline novel design axes (short-vs-long
# stream balance). Long-memory window W=36 (the paper's own example) held
# fixed -- more a "how far back does the deterministic compressor reach"
# constant than a capacity knob.
#
# Because this model is cheap to train (a small MLP, ~600 rows), tuning
# uses a real E=20-member ensemble per config (not a single seed) --
# learning from the FNO-DST lesson earlier this session that single-seed
# validation selection is unreliable.
# ============================================================================

source("dsno_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")   # n_train, n_valid, n_test, inner_train_frac

p <- ncol(A_K)
X_trainvalid_A <- A_K[seq_len(n_train + n_valid), , drop = FALSE]  # 574 months
TAU <- 6L
E_TUNE <- 20L
W_FIXED <- 36L

evaluate_config <- function(K, D) {
  B <- build_legendre_basis(W_FIXED, D)
  ds <- build_dual_stream(X_trainvalid_A, K, W_FIXED, D, TAU, B)
  n <- nrow(ds$X); n_tr <- floor(0.85 * n); idx_valid <- seq.int(n_tr + 1L, n)
  d_in <- p * (K + 1 + D)

  t0 <- Sys.time()
  fits <- fit_dsno_ensemble(ds$X, ds$Y, E = E_TUNE, d_in = d_in, p = p, seed0 = 1L,
                             epochs = 500L, patience = 20L, verbose_every = 501)
  fit_time <- as.numeric(Sys.time() - t0, units = "secs")

  arr_coef <- predict_dsno_ensemble_array(fits, ds$X[idx_valid, , drop = FALSE])  # (n_valid, p, E)
  n_valid <- length(idx_valid)
  pred_array_phys <- array(NA_real_, dim = c(n_valid, nrow(Psi_K), E_TUNE))
  for (e in seq_len(E_TUNE)) pred_array_phys[, , e] <- t(Psi_K %*% t(arr_coef[, , e]) + col_mean)
  truth_phys <- t(Psi_K %*% t(ds$Y[idx_valid, , drop = FALSE]) + col_mean)

  crps_val <- mean(crps_fair_array(pred_array_phys, truth_phys), na.rm = TRUE)
  mspe <- mean((apply(pred_array_phys, c(1, 2), mean) - truth_phys)^2)

  cat(sprintf("[K=%d D=%d] d_in=%d fit_time=%.1fs CRPS(valid)=%.4f mspe=%.4f\n",
              K, D, d_in, fit_time, crps_val, mspe))
  list(K = K, D = D, W = W_FIXED, d_in = d_in, fit_time = fit_time, crps_val = crps_val, mspe = mspe)
}

grid <- expand.grid(K = c(3L, 5L, 8L), D = c(2L, 4L, 6L))
cat(sprintf("\n================ Dual-Stream NO tuning grid: %d configs ================\n", nrow(grid)))
results <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  cat(sprintf("\n--- config %d/%d: K=%d D=%d ---\n", i, nrow(grid), grid$K[i], grid$D[i]))
  results[[i]] <- evaluate_config(grid$K[i], grid$D[i])
}

tab <- do.call(rbind, lapply(results, as.data.frame))
tab <- tab[order(tab$crps_val), ]
cat("\n================ Dual-Stream NO tuning results, sorted by validation CRPS ================\n")
print(tab, row.names = FALSE)
winner <- tab[1, ]
cat(sprintf("\nWinner: K=%d, D=%d, W=%d (CRPS=%.4f)\n", winner$K, winner$D, winner$W, winner$crps_val))

saveRDS(list(tab = tab, winner = winner), "dsno_tune_h6_results.rds")
cat("Saved dsno_tune_h6_results.rds\nDone.\n")
