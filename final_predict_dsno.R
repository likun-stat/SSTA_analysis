# ============================================================================
# final_predict_dsno.R
#
# Final Dual-Stream Neural Operator forecast for October 2026, ALL 676
# months, tuned winner (K, D; W=36), E=500 ensemble members (bumped up from
# the paper's own example E=100 to match every GRU variant's own 500-member-
# per-seed convention -- see dsno_fit_h6.R's header for why 500 independent
# networks, not 500 draws, is the faithful way to scale this model's
# ensemble). No padding trick needed -- this model only
# ever looks backward (W months of history), so a single feature vector at
# t=T (April 2026) is all that's required for a direct 6-month-ahead point
# forecast per ensemble member.
#
# Saves draws_pc (p x E, EOF-coefficient space) in the SAME format as
# final_predict_gru_iterated.R/final_predict_gru_direct.R, so it plugs
# directly into the existing final_plot_oct2026.R reconstruction pipeline.
# ============================================================================

source("dsno_h6.R")

tune <- readRDS("dsno_tune_h6_results.rds")
CFG <- list(K = tune$winner$K, D = tune$winner$D, W = 36L)
cat(sprintf("Using tuned config: K=%d, D=%d, W=%d\n", CFG$K, CFG$D, CFG$W))

p <- ncol(A_K)
TAU <- 6L
E_FINAL <- 500L

B <- build_legendre_basis(CFG$W, CFG$D)
ds <- build_dual_stream(A_K, CFG$K, CFG$W, CFG$D, TAU, B)  # ALL 676 months, no held-out test
d_in <- p * (CFG$K + 1 + CFG$D)
cat(sprintf("Training windows: %d, d_in=%d\n", nrow(ds$X), d_in))

t0 <- Sys.time()
fits <- fit_dsno_ensemble(ds$X, ds$Y, E = E_FINAL, d_in = d_in, p = p, seed0 = 1L,
                           epochs = 500L, patience = 20L, verbose_every = 501)
fit_time <- as.numeric(Sys.time() - t0, units = "mins")
cat(sprintf("Ensemble fit time: %.1f min\n", fit_time))

X_cur <- build_current_features(A_K, CFG$K, CFG$W, CFG$D, B)  # single row, t=T=April 2026
arr_coef <- predict_dsno_ensemble_array(fits, X_cur)  # (1, p, E)
draws_pc <- arr_coef[1, , ]  # (p, E) -- matches GRU scripts' saved format (dropping dim 1 already gives p x E; NO transpose)
cat(sprintf("Pooled draws: %d x %d\n", nrow(draws_pc), ncol(draws_pc)))

saveRDS(list(draws_pc = draws_pc, CFG = CFG, target_date = "2026-10-01"), "final_dsno_oct2026.rds")
cat("Saved final_dsno_oct2026.rds\nDone.\n")
