# ============================================================================
# fno_dstm2d_direct_h6.R
#
# FNO-DST (space-time Fourier neural operator + Gaussian-process-style
# conditional covariance) benchmark on the RAW SSTA field, forecasting 6
# months ahead (direct-h6, history=3) -- the comparison point for the
# EOF-reduced GRU jump-mixture model (eof_gru_fit_h6.R), since the original
# FNO-DST paper/repository does not use an EOF decomposition.
#
# Adapted from Lorentz-coupled/fno_dstm_direct_h3_benchmark.R, which was
# built for the 18-location Lorenz-96 ring. Two things there do not scale to
# the SST grid (m=2512 ocean cells of an 84x33 domain) and are replaced here:
#
#   1. Spatial structure: that file treats "space" as a 1D ordered index (a
#      periodic ring) and its spectral-conv layer FFTs over (that 1D axis,
#      history-time). Here spectral_conv3d FFTs over the genuine (lon, lat,
#      history-time) axes of the regular grid. Land cells are zero-filled
#      with an added ocean-mask input channel so the network can tell "land"
#      from "a true zero anomaly"; the loss is masked to ocean cells only.
#
#   2. Covariance: that file's Gaussian NLL forms a DENSE d x d covariance
#      via Cholesky every batch (fine at d=18) and predict_fno_dstm_direct
#      materializes an (n x d x d) array of them. At d=2512 that Cholesky
#      is 2512x2512 per sample and the prediction array alone would be tens
#      of GB -- not just slow, it does not fit in memory. Replaced with a
#      LOW-RANK-PLUS-DIAGONAL covariance: a real geographic (haversine) RBF
#      correlation matrix over the ocean cells, eigen-truncated to its top r
#      modes ONCE, then a Woodbury-identity Gaussian NLL that never
#      materializes a dense m x m matrix (O(m r^2) instead of O(m^3)).
#      Verified numerically identical (to float32 precision) to the original
#      dense-Cholesky formula on a small synthetic case with r = full rank
#      before being trusted at scale -- see the validation run in the
#      session transcript.
#
# Canonical spatial flatten order used THROUGHOUT this file: torch's own
# `torch_flatten(x, start_dim=2)` on a (batch, nlon, nlat) tensor, which
# empirically varies lat fastest: index k = (lon_i-1)*nlat + lat_j (1-based).
# This is DELIBERATELY NOT the same order eof_analysis.R uses (lon fastest,
# R's own as.vector() convention) -- fighting torch's native reshape
# semantics to match R's column-major order would require permute+reshape
# gymnastics on every tensor touching the ocean-cell subset; instead this
# file defines its own canonical order end-to-end (ocean_idx_t, geographic
# distances, lon/lat feature grids all built consistently with it in R), and
# a single explicit permutation vector (perm_t2r, built in section 7) is the
# only place the two conventions are reconciled, when projecting onto the
# EOF basis for the head-to-head comparison with the GRU-EOF model.
# ============================================================================

library(torch)
library(ncdf4)

set.seed(20260928)
torch_manual_seed(20260928L)

# ============================================================================
# 1. Load the raw SSTA grid (not EOF-reduced), build the land mask, and the
#    canonical (torch-native) flattened ocean index / geographic coordinates.
# ============================================================================
nc <- nc_open("sst_anomalies_extracted.nc")
sst <- ncvar_get(nc, "sst")   # (lon, lat, time)
lat <- ncvar_get(nc, "lat")
lon <- ncvar_get(nc, "lon")
nc_close(nc)

nlon <- length(lon); nlat <- length(lat); Tt <- dim(sst)[3]
m0 <- nlon * nlat

# X_grid_flat: (T, m0), canonical torch order (lat fastest within each lon
# block) -- built directly with that index formula, not via as.vector().
X_grid_flat <- matrix(NA_real_, nrow = Tt, ncol = m0)
idx_torch <- function(i, j) (i - 1L) * nlat + j   # lon_i, lat_j -> 1-based flat index
for (tt in seq_len(Tt)) {
  for (i in seq_len(nlon)) {
    X_grid_flat[tt, idx_torch(i, seq_len(nlat))] <- sst[i, , tt]
  }
}

ocean_mask_flat <- !apply(X_grid_flat, 2, function(v) all(is.na(v)))
stopifnot(!anyNA(X_grid_flat[, ocean_mask_flat]))  # land is all-NA, ocean is complete (checked in eof_analysis.R)
ocean_idx_t <- which(ocean_mask_flat)               # canonical-order ocean indices
m <- length(ocean_idx_t)
cat(sprintf("Grid: nlon=%d, nlat=%d, m0=%d, ocean m=%d\n", nlon, nlat, m0, m))

X_grid_flat[is.na(X_grid_flat)] <- 0   # zero-fill land

# lon/lat coordinate grids in the SAME canonical flatten order, for both the
# network's input features and the geographic distance matrix.
lon_flat <- numeric(m0); lat_flat <- numeric(m0)
for (i in seq_len(nlon)) { lon_flat[idx_torch(i, seq_len(nlat))] <- lon[i]; lat_flat[idx_torch(i, seq_len(nlat))] <- lat }

haversine_km <- function(lon1, lat1, lon2, lat2) {
  R_earth <- 6371
  p1 <- lat1 * pi / 180; p2 <- lat2 * pi / 180
  dphi <- (lat2 - lat1) * pi / 180
  dlmb <- (lon2 - lon1) * pi / 180
  a <- sin(dphi / 2)^2 + cos(p1) * cos(p2) * sin(dlmb / 2)^2
  2 * R_earth * asin(pmin(1, sqrt(a)))
}

# ============================================================================
# 2. Low-rank geographic RBF correlation kernel (ocean cells only), built and
#    eigen-truncated ONCE (fixed, non-trainable) -- see file header.
# ============================================================================
COV_LENGTHSCALE_KM <- 2500   # ENSO-scale spatial decorrelation length (large-scale tropical Pacific patterns)
COV_JITTER <- 0.1            # nugget, in standardized (unit-variance-target) units
RANK_R <- 40

cat("Building geographic distance matrix over ocean cells...\n")
lon_o <- lon_flat[ocean_idx_t]; lat_o <- lat_flat[ocean_idx_t]
dist_km <- outer(seq_len(m), seq_len(m), function(a, b) haversine_km(lon_o[a], lat_o[a], lon_o[b], lat_o[b]))
R_corr <- exp(-0.5 * (dist_km / COV_LENGTHSCALE_KM)^2)

cat("Eigen-decomposing (this is a one-time m x m cost)...\n")
t0 <- Sys.time()
eig <- eigen(R_corr, symmetric = TRUE)
cat(sprintf("  done in %.1fs\n", as.numeric(Sys.time() - t0, units = "secs")))
top_r <- seq_len(RANK_R)
Fbase_mat <- eig$vectors[, top_r, drop = FALSE] %*% diag(sqrt(pmax(eig$values[top_r], 0)))  # (m, r)
pve_kernel <- sum(pmax(eig$values[top_r], 0)) / sum(pmax(eig$values, 0)) * 100
cat(sprintf("Low-rank kernel: r=%d retains %.1f%% of the correlation matrix's trace\n", RANK_R, pve_kernel))

Fbase <- torch_tensor(Fbase_mat, dtype = torch_float())  # fixed buffer, (m, r)

# ============================================================================
# 3. Data helpers -- UNCHANGED from fno_dstm_direct_h3_benchmark.R (they only
#    ever treat "d" as a generic column count, so they apply verbatim to the
#    flattened m0-column grid).
# ============================================================================
make_chronological_split <- function(n, train_frac = 0.70, valid_frac = 0.15) {
  n_train <- floor(train_frac * n); n_valid <- floor(valid_frac * n); n_test <- n - n_train - n_valid
  if (min(n_train, n_valid, n_test) < 1L) stop("Not enough transition windows for the requested split.")
  list(train = seq_len(n_train), valid = seq.int(n_train + 1L, n_train + n_valid), test = seq.int(n_train + n_valid + 1L, n))
}

make_direct_windows <- function(X, history = 3L, horizon = 6L) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  n_time <- nrow(X); d <- ncol(X); n_window <- n_time - (history - 1L) - horizon
  if (n_window < 1L) stop("X is too short for the requested history and forecast horizon.")
  x <- array(NA_real_, dim = c(n_window, d, history)); y <- matrix(NA_real_, nrow = n_window, ncol = d)
  origin <- integer(n_window); target <- integer(n_window)
  for (i in seq_len(n_window)) {
    end_history <- i + history - 1L
    x[i, , ] <- t(X[i:end_history, , drop = FALSE])
    y[i, ] <- X[end_history + horizon, ]
    origin[i] <- end_history; target[i] <- end_history + horizon
  }
  list(x = x, y = y, origin = origin, target = target, n = n_window, d = d, history = history, horizon = horizon)
}

fit_coordinate_standardizer <- function(y_train) {
  center <- colMeans(y_train); scale <- apply(y_train, 2, stats::sd)
  scale[!is.finite(scale) | scale <= 0] <- 1   # land cells (always 0): scale->1, stays 0 after standardizing
  list(center = center, scale = scale)
}
standardize_windows <- function(x, standardizer) { out <- x; for (j in seq_len(dim(x)[2])) out[, j, ] <- (x[, j, ] - standardizer$center[j]) / standardizer$scale[j]; out }
standardize_targets <- function(y, standardizer) sweep(sweep(y, 2, standardizer$center, "-"), 2, standardizer$scale, "/")
invert_targets <- function(y, standardizer) sweep(sweep(y, 2, standardizer$scale, "*"), 2, standardizer$center, "+")

clone_state_dict <- function(module) lapply(module$state_dict(), function(x) x$detach()$clone())
set_module_trainable <- function(module, trainable = TRUE) { invisible(lapply(unname(module$parameters), function(p) p$requires_grad_(trainable))); invisible(module) }
count_parameters <- function(module) sum(vapply(unname(module$parameters), function(p) as.numeric(p$numel()), numeric(1)))

# ============================================================================
# 4. Spectral layers -- genuine 3-axis (lon, lat, history-time) FFT.
# ============================================================================
spectral_conv3d <- nn_module(
  "spectral_conv3d",
  initialize = function(in_channels, out_channels, modes_lon, modes_lat, modes_time) {
    self$in_channels <- as.integer(in_channels); self$out_channels <- as.integer(out_channels)
    self$modes_lon <- as.integer(modes_lon); self$modes_lat <- as.integer(modes_lat); self$modes_time <- as.integer(modes_time)
    init_scale <- 1 / (self$in_channels * self$out_channels)
    wdim <- c(self$in_channels, self$out_channels, self$modes_lon, self$modes_lat, self$modes_time)
    self$weight_real <- nn_parameter(init_scale * torch_rand(wdim, dtype = torch_float()))
    self$weight_imag <- nn_parameter(init_scale * torch_rand(wdim, dtype = torch_float()))
  },
  forward = function(x) {
    # x: [batch, channels, lon, lat, history]
    batch <- x$size(1); n_lon <- x$size(3); n_lat <- x$size(4); n_time <- x$size(5)
    n_time_fft <- floor(n_time / 2) + 1L
    if (self$modes_lon > n_lon || self$modes_lat > n_lat || self$modes_time > n_time_fft) {
      stop("Requested Fourier modes exceed the available grid frequencies.")
    }
    x_ft <- torch_fft_rfft(x, dim = 5L, norm = "ortho")
    x_ft <- torch_fft_fft(x_ft, dim = 4L, norm = "ortho")
    x_ft <- torch_fft_fft(x_ft, dim = 3L, norm = "ortho")

    x_low <- x_ft[, , seq_len(self$modes_lon), seq_len(self$modes_lat), seq_len(self$modes_time)]
    weight <- torch_complex(self$weight_real, self$weight_imag)
    low_product <- torch_sum(x_low$unsqueeze(3) * weight$unsqueeze(1), dim = 2L)  # sum over Cin

    if (self$modes_time < n_time_fft) {
      zero_t <- torch_zeros(c(batch, self$out_channels, self$modes_lon, self$modes_lat, n_time_fft - self$modes_time),
                             dtype = torch_cfloat(), device = x$device)
      low_product <- torch_cat(list(low_product, zero_t), dim = 5L)
    }
    if (self$modes_lat < n_lat) {
      zero_lat <- torch_zeros(c(batch, self$out_channels, self$modes_lon, n_lat - self$modes_lat, n_time_fft),
                               dtype = torch_cfloat(), device = x$device)
      low_product <- torch_cat(list(low_product, zero_lat), dim = 4L)
    }
    if (self$modes_lon < n_lon) {
      zero_lon <- torch_zeros(c(batch, self$out_channels, n_lon - self$modes_lon, n_lat, n_time_fft),
                               dtype = torch_cfloat(), device = x$device)
      out_ft <- torch_cat(list(low_product, zero_lon), dim = 3L)
    } else out_ft <- low_product

    out <- torch_fft_ifft(out_ft, n = n_lon, dim = 3L, norm = "ortho")
    out <- torch_fft_ifft(out, n = n_lat, dim = 4L, norm = "ortho")
    torch_fft_irfft(out, n = n_time, dim = 5L, norm = "ortho")
  }
)

fno_dstm_mean3d <- nn_module(
  "fno_dstm_mean3d",
  initialize = function(history, modes_lon, modes_lat, modes_time, width = 32L) {
    self$history <- as.integer(history); self$width <- as.integer(width)
    self$lift <- nn_linear(5L, self$width)   # value, ocean_mask, lon, lat, time
    for (b in 0:3) {
      self[[sprintf("spec%d", b)]] <- spectral_conv3d(self$width, self$width, modes_lon, modes_lat, modes_time)
      self[[sprintf("local%d", b)]] <- nn_conv3d(self$width, self$width, kernel_size = 1L)
      self[[sprintf("bn%d", b)]] <- nn_batch_norm3d(self$width)
    }
    self$project1 <- nn_linear(self$width, 128L)
    self$project2 <- nn_linear(128L, 1L)
    self$time_projection <- nn_linear(self$history, 1L)
  },
  forward = function(values, ocean_mask_2d) {
    # values: [batch, nlon, nlat, history]; ocean_mask_2d: [nlon, nlat] (0/1, shared)
    batch <- values$size(1); n_lon <- values$size(2); n_lat <- values$size(3); n_time <- values$size(4)

    lon_grid <- torch_linspace(0, 1 - 1 / n_lon, steps = n_lon, dtype = torch_float(), device = values$device)$view(c(1, n_lon, 1, 1, 1))$expand(c(batch, n_lon, n_lat, n_time, 1))
    lat_grid <- torch_linspace(0, 1 - 1 / n_lat, steps = n_lat, dtype = torch_float(), device = values$device)$view(c(1, 1, n_lat, 1, 1))$expand(c(batch, n_lon, n_lat, n_time, 1))
    time_grid <- torch_linspace(0, 1, steps = n_time, dtype = torch_float(), device = values$device)$view(c(1, 1, 1, n_time, 1))$expand(c(batch, n_lon, n_lat, n_time, 1))
    mask_grid <- ocean_mask_2d$view(c(1, n_lon, n_lat, 1, 1))$expand(c(batch, n_lon, n_lat, n_time, 1))

    x <- torch_cat(list(values$unsqueeze(5), mask_grid, lon_grid, lat_grid, time_grid), dim = 5L)
    x <- self$lift(x)$permute(c(1L, 5L, 2L, 3L, 4L))   # [B, width, lon, lat, time]

    for (b in 0:3) {
      spec <- self[[sprintf("spec%d", b)]]; local <- self[[sprintf("local%d", b)]]; bn <- self[[sprintf("bn%d", b)]]
      z <- bn(spec(x) + local(x))
      x <- if (b < 3) nnf_relu(z) else z
    }

    x <- x$permute(c(1L, 3L, 4L, 5L, 2L))   # [B, lon, lat, time, width]
    x <- nnf_relu(self$project1(x))
    x <- self$project2(x)$squeeze(5)         # [B, lon, lat, time]
    self$time_projection(x)$squeeze(4)       # [B, lon, lat]
  }
)

fno_dstm_scale2d <- nn_module(
  "fno_dstm_scale2d",
  initialize = function(history) {
    self$conv1 <- nn_conv2d(as.integer(history), 64L, kernel_size = 7L, padding = 3L)
    self$conv2 <- nn_conv2d(64L, 1L, kernel_size = 3L, padding = 1L)
  },
  forward = function(history_by_channel) {
    # history_by_channel: [batch, history, lon, lat]
    x <- nnf_relu(self$conv1(history_by_channel))
    raw <- nnf_softplus(self$conv2(x))$squeeze(2)   # [batch, lon, lat]
    torch_clamp(raw, min = 1e-4, max = 5)
  }
)

# ============================================================================
# 5. Low-rank-plus-diagonal Gaussian NLL (Woodbury identity) -- validated
#    against the dense-Cholesky formula on a small synthetic case; see the
#    file header.
# ============================================================================
gaussian_nll_lowrank <- function(y, mu, sd, Fbase, jitter) {
  # y, mu, sd: (batch, m); Fbase: (m, r) fixed, shared across the batch.
  mdim <- y$size(2); r <- Fbase$size(2)
  Fmat <- sd$unsqueeze(3) * Fbase$unsqueeze(1)         # (batch, m, r)
  e <- y - mu
  Ft <- Fmat$transpose(2, 3)                            # (batch, r, m)
  FtF <- torch_matmul(Ft, Fmat)                          # (batch, r, r)
  G <- torch_eye(r, dtype = y$dtype, device = y$device)$unsqueeze(1) + FtF / jitter
  cholG <- linalg_cholesky(G)
  b <- torch_matmul(Ft, e$unsqueeze(3))$squeeze(3)        # (batch, r)
  x <- torch_cholesky_solve(b$unsqueeze(3), cholG)$squeeze(3)
  term2 <- torch_sum(b * x, dim = 2L) / (jitter^2)
  quad <- torch_sum(e * e, dim = 2L) / jitter - term2
  log_det_G <- 2 * torch_sum(torch_log(torch_diagonal(cholG, dim1 = 2L, dim2 = 3L)), dim = 2L)
  log_det_sigma <- log_det_G + mdim * log(jitter)
  torch_mean(0.5 * (mdim * log(2 * pi) + log_det_sigma + quad))
}

cat("fno_dstm2d_direct_h6.R: setup complete (data, kernel, architecture, loss).\n")

# ============================================================================
# 6. Grid reshape helper (verified reshape recipe -- see session transcript):
#    flat (n, m0, ...) in canonical torch order -> (n, nlon, nlat, ...).
#    R's dim<- reassignment is column-major, so splitting m0 (built with lat
#    fastest) into (nlon, nlat) requires dim<-c(*, nlat, nlon, *) FIRST, then
#    an aperm() to swap those two axes into the intended (nlon, nlat) order.
# ============================================================================
reshape_flat_to_grid <- function(x_flat) {
  d <- dim(x_flat)
  if (length(d) == 2) {           # (n, m0) -> (n, nlon, nlat)
    dim(x_flat) <- c(d[1], nlat, nlon)
    aperm(x_flat, c(1, 3, 2))
  } else if (length(d) == 3) {    # (n, m0, history) -> (n, nlon, nlat, history)
    dim(x_flat) <- c(d[1], nlat, nlon, d[3])
    aperm(x_flat, c(1, 3, 2, 4))
  } else stop("reshape_flat_to_grid: expected a 2D or 3D array.")
}

ocean_mask_2d_mat <- matrix(as.numeric(ocean_mask_flat), nrow = nlat, ncol = nlon)
ocean_mask_2d_mat <- aperm(ocean_mask_2d_mat, c(2, 1))   # (nlon, nlat)
ocean_mask_2d <- torch_tensor(ocean_mask_2d_mat, dtype = torch_float())

# ============================================================================
# 7. Fitting -- same two-phase strategy (MSE pretrain, then joint
#    Gaussian-NLL refinement with periodic mean updates) as the original
#    file, adapted for: masked-to-ocean loss, low-rank covariance, and the
#    2D grid network.
# ============================================================================
fit_fno_dstm2d_direct <- function(
    X, history = 3L, horizon = 6L, modes_lon = 10L, modes_lat = 8L, modes_time = 2L,
    width = 32L, batch_size = 16L, learning_rate = 1e-3, weight_decay = 1e-4,
    lr_step_size = 10L, lr_gamma = 0.5, mean_epochs = 100L, nll_epochs = 50L,
    mean_patience = 12L, nll_patience = 30L, mean_update_every = 5L,
    train_frac = 0.70, valid_frac = 0.15, jitter = COV_JITTER, lambda_var = 10,
    seed = 123L, device = NULL) {

  torch_manual_seed(as.integer(seed)); set.seed(seed)
  if (is.null(device)) device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

  windows <- make_direct_windows(X, history = history, horizon = horizon)
  split <- make_chronological_split(windows$n, train_frac = train_frac, valid_frac = valid_frac)

  standardizer <- fit_coordinate_standardizer(windows$y[split$train, , drop = FALSE])
  x_std <- standardize_windows(windows$x, standardizer)   # (n, m0, history)
  y_std <- standardize_targets(windows$y, standardizer)   # (n, m0)

  x_grid <- reshape_flat_to_grid(x_std)   # (n, nlon, nlat, history)
  y_grid <- reshape_flat_to_grid(y_std)   # (n, nlon, nlat)

  make_loader <- function(idx, shuffle) {
    ds <- tensor_dataset(torch_tensor(x_grid[idx, , , , drop = FALSE], dtype = torch_float()),
                          torch_tensor(y_grid[idx, , , drop = FALSE], dtype = torch_float()))
    dataloader(ds, batch_size = batch_size, shuffle = shuffle)
  }
  train_loader <- make_loader(split$train, TRUE)
  valid_loader <- make_loader(split$valid, FALSE)

  mean_model <- fno_dstm_mean3d(history, modes_lon, modes_lat, modes_time, width)$to(device = device)
  scale_model <- fno_dstm_scale2d(history)$to(device = device)
  ocean_mask_dev <- ocean_mask_2d$to(device = device)
  Fbase_dev <- Fbase$to(device = device)

  cat(sprintf("Device: %s | windows: %d | grid: %dx%d (m0), ocean m=%d | train/valid/internal-test: %d/%d/%d\n",
              as.character(device), windows$n, nlon, nlat, m, length(split$train), length(split$valid), length(split$test)))
  cat(sprintf("FNO mean parameters: %d | scale-network parameters: %d\n",
              count_parameters(mean_model), count_parameters(scale_model)))

  flatten_ocean <- function(grid_tensor) torch_flatten(grid_tensor, start_dim = 2)[, ocean_idx_t]

  history_df <- data.frame(phase = character(), epoch = integer(),
                            train_mse = double(), valid_mse = double(), train_nll = double(), valid_nll = double())

  # ---- Phase 1: MSE pretraining (ocean cells only) ----
  mean_optimizer <- optim_adam(mean_model$parameters, lr = learning_rate, weight_decay = weight_decay)
  mean_scheduler <- lr_step(mean_optimizer, step_size = lr_step_size, gamma = lr_gamma)
  best_mean_state <- clone_state_dict(mean_model); best_valid_mse <- Inf; no_improvement <- 0L

  for (ep in seq_len(mean_epochs)) {
    mean_model$train(); train_sum <- 0; train_n <- 0L
    coro::loop(for (batch in train_loader) {
      xb <- batch[[1]]$to(device = device); yb <- batch[[2]]$to(device = device)
      mean_optimizer$zero_grad()
      mu <- mean_model(xb, ocean_mask_dev)
      loss <- nnf_mse_loss(flatten_ocean(mu), flatten_ocean(yb), reduction = "mean")
      loss$backward(); nn_utils_clip_grad_norm_(mean_model$parameters, max_norm = 5); mean_optimizer$step()
      bs <- xb$size(1); train_sum <- train_sum + loss$item() * bs; train_n <- train_n + bs
    })
    mean_model$eval(); valid_sum <- 0; valid_n <- 0L
    with_no_grad({
      coro::loop(for (batch in valid_loader) {
        xb <- batch[[1]]$to(device = device); yb <- batch[[2]]$to(device = device)
        loss <- nnf_mse_loss(flatten_ocean(mean_model(xb, ocean_mask_dev)), flatten_ocean(yb), reduction = "mean")
        bs <- xb$size(1); valid_sum <- valid_sum + loss$item() * bs; valid_n <- valid_n + bs
      })
    })
    train_mse <- train_sum / train_n; valid_mse <- valid_sum / valid_n
    history_df <- rbind(history_df, data.frame(phase = "mean_MSE", epoch = ep, train_mse = train_mse, valid_mse = valid_mse, train_nll = NA_real_, valid_nll = NA_real_))
    cat(sprintf("Mean epoch %03d | train_mse = %.6f | valid_mse = %.6f\n", ep, train_mse, valid_mse))
    if (valid_mse < best_valid_mse) { best_valid_mse <- valid_mse; best_mean_state <- clone_state_dict(mean_model); no_improvement <- 0L } else no_improvement <- no_improvement + 1L
    mean_scheduler$step()
    if (no_improvement >= mean_patience) { cat(sprintf("Mean-stage early stopping at epoch %d.\n", ep)); break }
  }
  mean_model$load_state_dict(best_mean_state)

  # ---- Phase 2: joint Gaussian-NLL refinement (low-rank, ocean cells only) ----
  combined_parameters <- c(unname(mean_model$parameters), unname(scale_model$parameters))
  joint_optimizer <- optim_adam(combined_parameters, lr = learning_rate, weight_decay = weight_decay)
  joint_scheduler <- lr_step(joint_optimizer, step_size = lr_step_size, gamma = lr_gamma)
  best_mean_nll_state <- clone_state_dict(mean_model); best_scale_state <- clone_state_dict(scale_model)
  best_valid_nll <- Inf; no_improvement <- 0L

  for (ep in seq_len(nll_epochs)) {
    update_mean <- ((ep - 1L) %% mean_update_every) == 0L
    set_module_trainable(mean_model, update_mean)
    if (update_mean) mean_model$train() else mean_model$eval()
    scale_model$train()

    train_nll_sum <- 0; train_mse_sum <- 0; train_n <- 0L; train_sd_sq_sum <- 0
    coro::loop(for (batch in train_loader) {
      xb <- batch[[1]]$to(device = device); yb <- batch[[2]]$to(device = device)
      joint_optimizer$zero_grad()
      mu <- mean_model(xb, ocean_mask_dev)
      sd_full <- scale_model(xb$permute(c(1, 4, 2, 3)))   # (batch, nlon, nlat)
      mu_o <- flatten_ocean(mu); y_o <- flatten_ocean(yb); sd_o <- flatten_ocean(sd_full)

      nll <- gaussian_nll_lowrank(y_o, mu_o, sd_o, Fbase_dev, jitter)
      var_penalty <- lambda_var * torch_mean(sd_o^2)
      loss <- nll + var_penalty
      loss$backward(); nn_utils_clip_grad_norm_(combined_parameters, max_norm = 5); joint_optimizer$step()

      mse <- nnf_mse_loss(mu_o, y_o, reduction = "mean")
      bs <- xb$size(1)
      train_nll_sum <- train_nll_sum + nll$item() * bs; train_mse_sum <- train_mse_sum + mse$item() * bs
      train_sd_sq_sum <- train_sd_sq_sum + mean(as.numeric(sd_o$detach()$cpu())^2) * bs; train_n <- train_n + bs
    })

    set_module_trainable(mean_model, TRUE); mean_model$eval(); scale_model$eval()
    valid_nll_sum <- 0; valid_mse_sum <- 0; valid_sd_sq_sum <- 0; valid_n <- 0L
    with_no_grad({
      coro::loop(for (batch in valid_loader) {
        xb <- batch[[1]]$to(device = device); yb <- batch[[2]]$to(device = device)
        mu <- mean_model(xb, ocean_mask_dev)
        sd_full <- scale_model(xb$permute(c(1, 4, 2, 3)))
        mu_o <- flatten_ocean(mu); y_o <- flatten_ocean(yb); sd_o <- flatten_ocean(sd_full)
        nll <- gaussian_nll_lowrank(y_o, mu_o, sd_o, Fbase_dev, jitter)
        mse <- nnf_mse_loss(mu_o, y_o, reduction = "mean")
        bs <- xb$size(1)
        valid_nll_sum <- valid_nll_sum + nll$item() * bs; valid_mse_sum <- valid_mse_sum + mse$item() * bs
        valid_sd_sq_sum <- valid_sd_sq_sum + mean(as.numeric(sd_o$cpu())^2) * bs; valid_n <- valid_n + bs
      })
    })

    train_nll <- train_nll_sum / train_n; valid_nll <- valid_nll_sum / valid_n
    train_mse <- train_mse_sum / train_n; valid_mse <- valid_mse_sum / valid_n
    history_df <- rbind(history_df, data.frame(phase = "Gaussian_NLL", epoch = ep, train_mse = train_mse, valid_mse = valid_mse, train_nll = train_nll, valid_nll = valid_nll))
    cat(sprintf("NLL epoch %03d | update_mean = %s | train_nll = %.4f | valid_nll = %.4f | valid_mse = %.6f | mean(sd^2) train/valid = %.4f/%.4f\n",
                ep, update_mean, train_nll, valid_nll, valid_mse, train_sd_sq_sum / train_n, valid_sd_sq_sum / valid_n))
    if (valid_nll < best_valid_nll) { best_valid_nll <- valid_nll; best_mean_nll_state <- clone_state_dict(mean_model); best_scale_state <- clone_state_dict(scale_model); no_improvement <- 0L } else no_improvement <- no_improvement + 1L
    joint_scheduler$step()
    if (no_improvement >= nll_patience) { cat(sprintf("NLL-stage early stopping at epoch %d.\n", ep)); break }
  }

  set_module_trainable(mean_model, TRUE)
  mean_model$load_state_dict(best_mean_nll_state); scale_model$load_state_dict(best_scale_state)
  mean_model$eval(); scale_model$eval()

  list(mean_model = mean_model, scale_model = scale_model, history = history_df, standardizer = standardizer,
       device = device, split = split, windows = windows, history_length = history, forecast_horizon = horizon,
       jitter = jitter, best_valid_mse = best_valid_mse, best_valid_nll = best_valid_nll, seed = seed)
}

# ============================================================================
# 8. Prediction: ensemble draws from the low-rank Gaussian (never
#    materializes a dense covariance) -- mean + F z + sqrt(jitter) eps, z ~
#    N(0, I_r), eps ~ N(0, I_m). Ocean cells only.
# ============================================================================
predict_fno_dstm2d_direct <- function(fit, X_test, n_ens = 200L, seed = NULL, batch_size = 32L) {
  if (!is.null(seed)) { set.seed(seed); torch_manual_seed(as.integer(seed)) }
  test_windows <- make_direct_windows(X_test, history = fit$history_length, horizon = fit$forecast_horizon)
  x_std <- standardize_windows(test_windows$x, fit$standardizer)
  x_grid <- reshape_flat_to_grid(x_std)

  n <- test_windows$n
  mu_std <- matrix(NA_real_, n, m); sd_std <- matrix(NA_real_, n, m)
  ocean_mask_dev <- ocean_mask_2d$to(device = fit$device)

  fit$mean_model$eval(); fit$scale_model$eval()
  with_no_grad({
    cursor <- 1L
    while (cursor <= n) {
      rows <- seq.int(cursor, min(cursor + batch_size - 1L, n))
      xb <- torch_tensor(x_grid[rows, , , , drop = FALSE], dtype = torch_float())$to(device = fit$device)
      mu <- fit$mean_model(xb, ocean_mask_dev)
      sd_full <- fit$scale_model(xb$permute(c(1, 4, 2, 3)))
      mu_std[rows, ] <- as_array(torch_flatten(mu, start_dim = 2)[, ocean_idx_t]$cpu())
      sd_std[rows, ] <- as_array(torch_flatten(sd_full, start_dim = 2)[, ocean_idx_t]$cpu())
      cursor <- cursor + batch_size
    }
  })

  # truth (ocean cells, original scale)
  truth_o <- test_windows$y[, ocean_idx_t, drop = FALSE]
  mu_o_original <- invert_targets(mu_std, list(center = fit$standardizer$center[ocean_idx_t], scale = fit$standardizer$scale[ocean_idx_t]))

  # Ensemble draws in STANDARDIZED units, then destandardize (mirrors the
  # GRU-jump model's own convention: pred_array on the same raw scale used
  # for scoring throughout this project).
  Fbase_np <- as.matrix(Fbase)  # (m, r)
  r <- ncol(Fbase_np)
  pred_array <- array(NA_real_, dim = c(n, m, n_ens))
  for (i in seq_len(n)) {
    Z <- matrix(rnorm(r * n_ens), r, n_ens)
    eps_shared <- Fbase_np %*% Z                              # (m, n_ens), shared low-rank part
    eps_diag <- matrix(rnorm(m * n_ens), m, n_ens) * sqrt(fit$jitter)
    draws_std <- mu_std[i, ] + sd_std[i, ] * (eps_shared + eps_diag)  # (m, n_ens)
    pred_array[i, , ] <- draws_std * fit$standardizer$scale[ocean_idx_t] + fit$standardizer$center[ocean_idx_t]
  }

  list(pred_array = pred_array, truth = truth_o, pred_mean = mu_o_original,
       mu_std = mu_std, sd_std = sd_std, origin = test_windows$origin, target = test_windows$target,
       lead = fit$forecast_horizon)
}

cat("fit_fno_dstm2d_direct / predict_fno_dstm2d_direct defined.\n")
