"""Small, transparent utilities used by the PINN teaching notebooks.

The implementation deliberately uses ``autograd`` rather than a large deep-learning
framework.  Every parameter lives in one vector, so the relation between the
mathematics and the code remains visible.  These routines are for teaching and
small experiments, not production-scale training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import autograd.numpy as anp
import numpy as np
from autograd import grad


Array = anp.ndarray


def set_plot_style() -> None:
    """Apply a consistent, accessible Matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.2),
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def _parameter_count(layers: Sequence[int]) -> int:
    return sum((din + 1) * dout for din, dout in zip(layers[:-1], layers[1:]))


def init_mlp(layers: Sequence[int], seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Xavier-initialize a fully connected tanh MLP into one flat vector."""
    rng = np.random.default_rng(seed)
    pieces: List[np.ndarray] = []
    for din, dout in zip(layers[:-1], layers[1:]):
        limit = scale * np.sqrt(6.0 / (din + dout))
        pieces.append(rng.uniform(-limit, limit, size=(din, dout)).ravel())
        pieces.append(np.zeros(dout))
    theta = np.concatenate(pieces)
    assert theta.size == _parameter_count(layers)
    return theta


def unpack_mlp(theta: Array, layers: Sequence[int]) -> List[Tuple[Array, Array]]:
    """View a flat parameter vector as weight/bias pairs."""
    params: List[Tuple[Array, Array]] = []
    cursor = 0
    for din, dout in zip(layers[:-1], layers[1:]):
        n_w = din * dout
        w = anp.reshape(theta[cursor : cursor + n_w], (din, dout))
        cursor += n_w
        b = theta[cursor : cursor + dout]
        cursor += dout
        params.append((w, b))
    return params


def mlp(theta: Array, x: Array, layers: Sequence[int]) -> Array:
    """Evaluate a tanh MLP. ``x`` has shape ``(batch, input_dimension)``."""
    h = x
    params = unpack_mlp(theta, layers)
    for w, b in params[:-1]:
        h = anp.tanh(anp.dot(h, w) + b)
    w_last, b_last = params[-1]
    return anp.dot(h, w_last) + b_last


@dataclass
class TrainResult:
    theta: np.ndarray
    history: Dict[str, List[float]]
    elapsed_seconds: float


def adam(
    loss_fn: Callable[[Array], Array],
    theta0: np.ndarray,
    steps: int = 1_000,
    learning_rate: float = 1e-3,
    callback: Callable[[int, np.ndarray, Dict[str, float]], Dict[str, float]] | None = None,
    log_every: int = 50,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> TrainResult:
    """Minimal full-batch Adam with history collection."""
    value_and_grad = grad(loss_fn)
    theta = np.asarray(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history: Dict[str, List[float]] = {"step": [], "loss": []}
    started = time.perf_counter()

    for step in range(1, steps + 1):
        g = np.asarray(value_and_grad(theta), dtype=float)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        should_log = step == 1 or step % log_every == 0 or step == steps
        if should_log:
            metrics: Dict[str, float] = {"loss": float(loss_fn(theta))}
            if callback is not None:
                metrics.update(callback(step, theta, dict(metrics)))
            history["step"].append(step)
            for key, value in metrics.items():
                history.setdefault(key, []).append(float(value))

    return TrainResult(theta, history, time.perf_counter() - started)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Discrete relative L2 error with a zero-reference safeguard."""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((prediction - reference).ravel()) / max(denominator, 1e-14))


def gradient_cosine(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cosine between two gradients; negative values indicate local conflict."""
    denom = np.linalg.norm(g1) * np.linalg.norm(g2)
    return float(np.dot(g1, g2) / max(denom, 1e-14))


def latin_hypercube_1d(n: int, low: float, high: float, seed: int = 0) -> np.ndarray:
    """A tiny one-dimensional Latin-hypercube sampler."""
    rng = np.random.default_rng(seed)
    points = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(points)
    return low + (high - low) * points

