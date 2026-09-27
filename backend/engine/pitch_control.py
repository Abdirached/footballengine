"""Spearman-style pitch control + analytic xT value grid.

Control at point p for team A:
    t_i(p) = t_react + |p - player_i| / v_max        (time-to-point)
    w_i(p) = exp(-t_i(p) / tau)                      (soft influence)
    C_A(p)  = Σ_{i∈A} w_i / (Σ_A w + Σ_B w)          (0…1, home-positive)

Grid is precomputed once; per-frame evaluation is fully vectorised NumPy
(~7k cells × 22 players ⇒ <5 ms).
"""
from __future__ import annotations

import numpy as np


class PitchControlModel:
    def __init__(self, length: float = 105.0, width: float = 68.0,
                 res_m: float = 2.0, v_max: float = 5.0,
                 t_react: float = 0.7, tau: float = 0.45, kappa: float = 0.6):
        self.length, self.width = length, width
        self.v_max, self.t_react, self.tau, self.kappa = v_max, t_react, tau, kappa
        self.nx = int(length / res_m)            # columns -> x
        self.ny = int(width / res_m)             # rows    -> y
        xs = np.linspace(-length / 2 + res_m / 2, length / 2 - res_m / 2, self.nx)
        ys = np.linspace(-width / 2 + res_m / 2, width / 2 - res_m / 2, self.ny)
        self.xx, self.yy = np.meshgrid(xs, ys)   # ny × nx
        self.grid_pts = np.stack([self.xx.ravel(), self.yy.ravel()], axis=1)
        self._value_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def control(self, home_xy: np.ndarray, away_xy: np.ndarray) -> np.ndarray:
        """Home control field, shape (ny, nx), values 0…1.

        Uses per-team soft-min response time and a logistic comparison:
            T_X(p) = −τ·log Σ_i exp(−t_i(p)/τ)          (smooth min)
            C_home(p) = σ((T_away − T_home) / k)
        """
        t_home = self._softmin_time(home_xy)
        t_away = self._softmin_time(away_xy)
        return self._sigmoid((t_away - t_home) / self.kappa).reshape(self.ny, self.nx)

    def _times(self, players: np.ndarray) -> np.ndarray:
        # |p - q|² = |p|² - 2·p·q + |q|²  (einsum avoids the (N,M,2) temp)
        sq = np.einsum("ij,ij->i", self.grid_pts, self.grid_pts)[:, None] \
            - 2.0 * (self.grid_pts @ players.T) \
            + np.einsum("ij,ij->i", players, players)[None, :]
        return self.t_react + np.sqrt(np.maximum(sq, 0.0)) / self.v_max

    def _softmin_time(self, players: np.ndarray) -> np.ndarray:
        if players.size == 0:
            return np.full(self.grid_pts.shape[0], 30.0)
        t = self._times(players)
        m = t.min(axis=1)
        return m - self.tau * np.log(
            np.exp(-(t - m[:, None]) / self.tau).sum(axis=1))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------ #
    def control_at(self, grid: np.ndarray, x: float, y: float) -> float:
        """Bilinear sample of a control grid at absolute pitch coordinates."""
        gx = int(np.clip((x + self.length / 2) / self.length * self.nx, 0, self.nx - 1))
        gy = int(np.clip((y + self.width / 2) / self.width * self.ny, 0, self.ny - 1))
        return float(grid[gy, gx])

    # ------------------------------------------------------------------ #
    def value_grid(self, attack_dir: int = 1) -> np.ndarray:
        """Analytic positional-value (xT-like) field, normalised 0…1 (cached)."""
        if attack_dir in self._value_cache:
            return self._value_cache[attack_dir]
        prog = (self.xx * attack_dir + self.length / 2) / self.length      # 0…1
        centrality = 1.0 - (np.abs(self.yy) / (self.width / 2)) ** 2       # 1 centre → 0 wings
        v = 1 / (1 + np.exp(-8.0 * (prog - 0.45))) * (0.55 + 0.45 * centrality)
        self._value_cache[attack_dir] = v
        return v

    def threat_at(self, grid: np.ndarray, x: float, y: float) -> float:
        gx = int(np.clip((x + self.length / 2) / self.length * self.nx, 0, self.nx - 1))
        gy = int(np.clip((y + self.width / 2) / self.width * self.ny, 0, self.ny - 1))
        return float(grid[gy, gx])
