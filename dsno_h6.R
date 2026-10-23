# ============================================================================
# dsno_h6.R
#
# Faithful implementation of C.K. Wikle's "Dual-Stream Hybrid Empirical
# Neural Operator" (Chris_model_description-v1.pdf), adapted to our K=27
# EOF basis (Section 2 of the write-up) rather than the paper's own
# illustrative p=10 -- for consistency with the other EOF-based models in
# this study, which all use the SAME permutation-scree-selected K=27 basis.
#
# Architecture (paper Eq 4-9): PC series alpha_t in R^p (p=27, our EOFs).
# Dual-stream feature vector at time t:
#   x_short(t) = vec([alpha_t, alpha_{t-1}, ..., alpha_{t-K}])   in R^{p(K+1)}
#   x_long(t)  = vec(B %*% A_t), A_t = [alpha_{t-W+1},...,alpha_t]^T in R^{W x p},
#                B in R^{D x W} a row-normalized discrete Legendre-polynomial
#                basis (Bonnet's recursion, Eq 5-6)               in R^{pD}
#   X(t) = [x_short(t); x_long(t)]                                in R^{p(K+1+D)}
# Model: MLP M_theta: X(t) -> alpha_hat_{t+tau}, trained by plain MSE (Eq 10)
# -- NO innovation/probabilistic head, exactly as the paper specifies
# ("we don't have an innovation error term here").
#
# Uncertainty (Sec 4.1): an ensemble of E independently-initialized networks;
# each gives a deterministic point forecast, and ensemble spread across the
# E members is the ONLY source of predictive uncertainty (a "deep ensemble"
# / epistemic-only UQ mechanism -- structurally different from every other
# model in this study, all of which have an explicit stochastic/generative
# output). For CRPS-style scoring we treat the E reconstructed physical-
# space point forecasts directly as the ensemble sample, exactly as the
# paper's own Eq 11-12 (ensemble mean/sd) already does implicitly.
# ============================================================================

load("eof_results.RData")   # A_K (T x p), Psi_K, col_mean, K_selected

# ------------------------------------------------------------------
# 1. Discrete Legendre basis (Bonnet's recursion, Eq 5-6). Validated against
#    the paper's formulas in the session transcript (row-normalized; only
#    approximately discretely-orthogonal across degrees at finite W, since
#    it's the continuous recursion sampled at equally-spaced points rather
#    than a true discrete/Gram-polynomial construction -- implemented
#    faithfully as the paper specifies, not "corrected").
# ------------------------------------------------------------------
build_legendre_basis <- function(W, D) {
  xi <- 2 * (0:(W - 1)) / (W - 1) - 1
  P <- matrix(NA_real_, D, W)
  P[1, ] <- 1
  if (D > 1) P[2, ] <- xi
  if (D > 2) for (n in 1:(D - 2)) P[n + 2, ] <- ((2 * n + 1) * xi * P[n + 1, ] - n * P[n, ]) / (n + 1)
  P / sqrt(rowSums(P^2))
}

# ------------------------------------------------------------------
# 2. Dual-stream feature construction (Eq 4, 7, 8). Returns X (n x d_in),
#    Y (n x p) for all valid training origins t in [W, T-tau], plus a
#    single-row X for the CURRENT time (t=T, no target -- used for the
#    Oct-2026 forecast, no padding trick needed since this model needs no
#    future data at all, only history).
# ------------------------------------------------------------------
build_dual_stream <- function(A, K, W, D, tau, B) {
  Tt <- nrow(A); p <- ncol(A)
  t_min <- W; t_max <- Tt - tau
  n <- t_max - t_min + 1
  d_in <- p * (K + 1 + D)
  X <- matrix(NA_real_, n, d_in)
  Y <- matrix(NA_real_, n, p)
  for (i in seq_len(n)) {
    t <- t_min + i - 1
    x_short <- as.vector(t(A[(t - K):t, , drop = FALSE]))          # (K+1)*p, alpha_t first
    A_t <- A[(t - W + 1):t, , drop = FALSE]                         # W x p
    C_t <- B %*% A_t                                                # D x p
    x_long <- as.vector(t(C_t))                                     # D*p
    X[i, ] <- c(x_short, x_long)
    Y[i, ] <- A[t + tau, ]
  }
  list(X = X, Y = Y, t = seq(t_min, t_max))
}

# Features/targets for an ARBITRARY vector of absolute origin indices t
# (not necessarily contiguous) -- used to score a fixed, exact set of test
# target months matching every other model's own 94-origin test window
# (target rows 583:676 of the full 676-month series), regardless of this
# model's own W/K/tau-driven natural windowing.
build_features_for_origins <- function(A, origins, K, W, D, B, tau) {
  p <- ncol(A); n <- length(origins); d_in <- p * (K + 1 + D)
  X <- matrix(NA_real_, n, d_in); Y <- matrix(NA_real_, n, p)
  for (i in seq_len(n)) {
    t <- origins[i]
    x_short <- as.vector(t(A[(t - K):t, , drop = FALSE]))
    A_t <- A[(t - W + 1):t, , drop = FALSE]
    C_t <- B %*% A_t
    x_long <- as.vector(t(C_t))
    X[i, ] <- c(x_short, x_long)
    Y[i, ] <- A[t + tau, ]
  }
  list(X = X, Y = Y)
}

build_current_features <- function(A, K, W, D, B) {
  Tt <- nrow(A); p <- ncol(A)
  t <- Tt
  x_short <- as.vector(t(A[(t - K):t, , drop = FALSE]))
  A_t <- A[(t - W + 1):t, , drop = FALSE]
  C_t <- B %*% A_t
  x_long <- as.vector(t(C_t))
  matrix(c(x_short, x_long), nrow = 1)
}

# ------------------------------------------------------------------
# 3. MLP M_theta (Eq 9): 2 hidden layers, fixed architecture (not part of
#    the tuning grid -- K and D are the paper's own headline novel design
#    axes and are what we tune, mirroring how the other models' tuning
#    grids targeted each paper's own central architectural choices).
# ------------------------------------------------------------------
library(torch)

dsno_mlp <- nn_module(
  "dsno_mlp",
  initialize = function(d_in, p, hidden1 = 128L, hidden2 = 64L, dropout = 0.1) {
    self$fc1 <- nn_linear(d_in, hidden1)
    self$fc2 <- nn_linear(hidden1, hidden2)
    self$fc3 <- nn_linear(hidden2, p)
    self$drop <- nn_dropout(p = dropout)
  },
  forward = function(x) {
    x <- self$drop(nnf_relu(self$fc1(x)))
    x <- self$drop(nnf_relu(self$fc2(x)))
    self$fc3(x)
  }
)

# ------------------------------------------------------------------
# 4. Fit one network: plain MSE (Eq 10), chronological train/valid split,
#    Adam + LR-on-plateau + early stopping (matching this project's
#    established convention for every other model, even though the paper
#    itself does not specify a validation/early-stopping procedure).
# ------------------------------------------------------------------
fit_dsno_one <- function(X, Y, train_frac = 0.85, lr = 1e-3, epochs = 500L,
                          batch_size = 32L, patience = 20L, min_delta = 1e-5,
                          weight_decay = 1e-4, seed = 1L, d_in, p, verbose_every = 501) {
  torch_manual_seed(as.integer(seed)); set.seed(seed)
  n <- nrow(X); n_train <- floor(train_frac * n)
  idx_train <- seq_len(n_train); idx_valid <- seq.int(n_train + 1L, n)

  center_x <- colMeans(X[idx_train, , drop = FALSE]); scale_x <- apply(X[idx_train, , drop = FALSE], 2, sd)
  scale_x[!is.finite(scale_x) | scale_x <= 0] <- 1
  center_y <- colMeans(Y[idx_train, , drop = FALSE]); scale_y <- apply(Y[idx_train, , drop = FALSE], 2, sd)
  scale_y[!is.finite(scale_y) | scale_y <= 0] <- 1

  Xs <- sweep(sweep(X, 2, center_x, "-"), 2, scale_x, "/")
  Ys <- sweep(sweep(Y, 2, center_y, "-"), 2, scale_y, "/")

  x_train_t <- torch_tensor(Xs[idx_train, , drop = FALSE], dtype = torch_float())
  y_train_t <- torch_tensor(Ys[idx_train, , drop = FALSE], dtype = torch_float())
  x_valid_t <- torch_tensor(Xs[idx_valid, , drop = FALSE], dtype = torch_float())
  y_valid_t <- torch_tensor(Ys[idx_valid, , drop = FALSE], dtype = torch_float())

  net <- dsno_mlp(d_in, p)
  opt <- optim_adam(net$parameters, lr = lr, weight_decay = weight_decay)
  scheduler <- lr_reduce_on_plateau(opt, mode = "min", factor = 0.5, patience = 8, min_lr = 1e-6)

  n_tr <- length(idx_train)
  best_valid <- Inf; best_state <- NULL; no_improve <- 0L; final_epoch <- 0L
  for (ep in seq_len(epochs)) {
    net$train()
    perm <- sample.int(n_tr)
    for (b0 in seq(1, n_tr, by = batch_size)) {
      idx <- perm[b0:min(b0 + batch_size - 1L, n_tr)]
      opt$zero_grad()
      pred <- net(x_train_t[idx, ])
      loss <- nnf_mse_loss(pred, y_train_t[idx, ])
      loss$backward(); opt$step()
    }
    net$eval()
    valid_loss <- with_no_grad({ nnf_mse_loss(net(x_valid_t), y_valid_t)$item() })
    final_epoch <- ep
    if (ep == 1L || ep %% verbose_every == 0L) cat(sprintf("  epoch %d valid_mse=%.5f\n", ep, valid_loss))
    if (valid_loss < best_valid - min_delta) { best_valid <- valid_loss; best_state <- lapply(net$state_dict(), function(z) z$detach()$clone()); no_improve <- 0L } else no_improve <- no_improve + 1L
    scheduler$step(valid_loss)
    if (no_improve >= patience) break
  }
  if (!is.null(best_state)) net$load_state_dict(best_state)
  net$eval()
  list(net = net, center_x = center_x, scale_x = scale_x, center_y = center_y, scale_y = scale_y,
       best_valid = best_valid, final_epoch = final_epoch)
}

predict_dsno_one <- function(fit, X_new) {
  Xs <- sweep(sweep(X_new, 2, fit$center_x, "-"), 2, fit$scale_x, "/")
  with_no_grad({
    pred_std <- as.matrix(fit$net(torch_tensor(Xs, dtype = torch_float())))
  })
  sweep(sweep(pred_std, 2, fit$scale_y, "*"), 2, fit$center_y, "+")
}

# ------------------------------------------------------------------
# 5. Ensemble wrapper: E independently-seeded networks (Sec 4.1). Returns a
#    function that predicts an (n_origin x p x E) array of EOF-coefficient
#    point forecasts for any feature matrix (one row per origin).
# ------------------------------------------------------------------
fit_dsno_ensemble <- function(X, Y, E, d_in, p, seed0 = 1L, ...) {
  fits <- vector("list", E)
  for (e in seq_len(E)) fits[[e]] <- fit_dsno_one(X, Y, seed = seed0 + e - 1L, d_in = d_in, p = p, ...)
  fits
}

predict_dsno_ensemble_array <- function(fits, X_new) {
  E <- length(fits); n <- nrow(X_new); p <- length(fits[[1]]$center_y)
  arr <- array(NA_real_, dim = c(n, p, E))
  for (e in seq_len(E)) arr[, , e] <- predict_dsno_one(fits[[e]], X_new)
  arr
}

cat("dsno_h6.R: Dual-Stream Neural Operator pipeline defined.\n")
