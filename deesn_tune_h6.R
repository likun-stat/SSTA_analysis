# ============================================================================
# deesn_tune_h6.R
#
# Hyperparameter grid search for D-EESN, matching the same 9-config,
# validation-CRPS-selected pattern used for every other model in this study.
# Grid: network depth numLayers in {3,5,7} x EOF-bottleneck width
# layerOneDimRed in {5,10,15} -- the paper's own headline novel design axes
# (how deep the hierarchical reservoir stack is, and how much each layer
# compresses before handing off to the next). Reservoir width, sparsity,
# weight scale and ridge penalty held fixed at generic paper defaults (see
# deesn_eof_h6.R header) -- domain-agnostic "how big is a reservoir"
# constants, not this paper's capacity knobs.
#
# Because a reservoir fit here is pure linear algebra (no gradient descent --
# random weights + one ridge-regression solve per member), tuning uses a
# genuine E=20-member ensemble per config, same as DSNO/CNN-IDE/GRU.
#
# Validation split: internal 85/15 split of the SAME 574-month train+valid
# block used everywhere else (X_trainvalid_A), by origin-month index
# (chronological). Reconstruction + CRPS use the EOF-restricted truth
# (Psi_K %*% true coefs + col_mean) as a validation-only proxy for physical
# space -- same simplification dsno_tune_h6.R uses; final reported numbers
# use the true NetCDF-reconstructed physical truth (deesn_fit_h6.R).
# ============================================================================

source("deesn_eof_h6.R")
source("eval_crps_utils.R")
load("eof_gru_data.RData")   # n_train, n_valid

p <- ncol(A_K)
X_trainvalid_A <- A_K[seq_len(n_train + n_valid), , drop = FALSE]  # 574 months
TAU <- 6L
TAU_EMBED <- TAU   # embedding lag = lead (paper's own convention)
E_TUNE <- 20L
n_local <- nrow(X_trainvalid_A)

all_origins <- seq.int(M_EMBED * TAU_EMBED + 1L, n_local - TAU)
n_all <- length(all_origins)
n_tr_local <- floor(0.85 * n_all)
train_origins <- all_origins[seq_len(n_tr_local)]
valid_origins <- all_origins[(n_tr_local + 1L):n_all]
train_target_end <- max(train_origins) + TAU

cat(sprintf("Local origins: %d total, %d train, %d valid\n", n_all, length(train_origins), length(valid_origins)))

truth_coef_valid <- X_trainvalid_A[valid_origins + TAU, , drop = FALSE]
truth_phys_valid <- t(Psi_K %*% t(truth_coef_valid) + col_mean)

evaluate_config <- function(numLayers, layerOneDimRed) {
  des <- build_deesn_design(X_trainvalid_A, train_target_end, valid_origins, TAU_EMBED, m = M_EMBED)

  t0 <- Sys.time()
  arr_coef <- fit_deesn_ensemble(des, p = p, numLayers = numLayers, layerOneDimRed = layerOneDimRed,
                                  E = E_TUNE, seed0 = 1L)   # (n_valid, p, E)
  fit_time <- as.numeric(Sys.time() - t0, units = "secs")

  n_valid <- dim(arr_coef)[1]
  pred_array_phys <- array(NA_real_, dim = c(n_valid, nrow(Psi_K), E_TUNE))
  for (e in seq_len(E_TUNE)) pred_array_phys[, , e] <- t(Psi_K %*% t(arr_coef[, , e]) + col_mean)

  crps_val <- mean(crps_fair_array(pred_array_phys, truth_phys_valid), na.rm = TRUE)
  mspe <- mean((apply(pred_array_phys, c(1, 2), mean) - truth_phys_valid)^2)

  cat(sprintf("[numLayers=%d layerOneDimRed=%d] fit_time=%.1fs CRPS(valid)=%.4f mspe=%.4f\n",
              numLayers, layerOneDimRed, fit_time, crps_val, mspe))
  list(numLayers = numLayers, layerOneDimRed = layerOneDimRed, fit_time = fit_time, crps_val = crps_val, mspe = mspe)
}

grid <- expand.grid(numLayers = c(3L, 5L, 7L), layerOneDimRed = c(5L, 10L, 15L))
cat(sprintf("\n================ D-EESN tuning grid: %d configs ================\n", nrow(grid)))
results <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  cat(sprintf("\n--- config %d/%d: numLayers=%d layerOneDimRed=%d ---\n", i, nrow(grid), grid$numLayers[i], grid$layerOneDimRed[i]))
  results[[i]] <- evaluate_config(grid$numLayers[i], grid$layerOneDimRed[i])
}

tab <- do.call(rbind, lapply(results, as.data.frame))
tab <- tab[order(tab$crps_val), ]
cat("\n================ D-EESN tuning results, sorted by validation CRPS ================\n")
print(tab, row.names = FALSE)
winner <- tab[1, ]
cat(sprintf("\nWinner: numLayers=%d, layerOneDimRed=%d (CRPS=%.4f)\n", winner$numLayers, winner$layerOneDimRed, winner$crps_val))

saveRDS(list(tab = tab, winner = winner), "deesn_tune_h6_results.rds")
cat("Saved deesn_tune_h6_results.rds\nDone.\n")
