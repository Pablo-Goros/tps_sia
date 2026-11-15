"""Variational autoencoder for TP5 exercise 2, built on TP3 shared pieces.

Activations, the Xavier rule and the optimizers come from ``tp3.shared``.
The forward and backward passes live here because the TP3 MLP trains end to
end and has no stochastic layer (mu, log-variance, reparameterization, KL).
"""
from pathlib import Path
import json
import sys

import numpy as np

# Resolve the sibling package once, independent of the caller's directory.
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tp3.shared.activations import construir_activacion
from tp3.shared.optimizers import construir_optimizador


def _widths(values, name):
    values = list(values)
    if len(values) < 2 or any(type(v) is not int or v < 1 for v in values):
        raise ValueError(f'{name} must list at least two positive integer widths')
    return values


class VAE:
    """Encoder -> (mu, log sigma^2) -> z = mu + sigma * eps -> decoder.

    Loss per batch (mean over samples): BCE summed over pixels plus
    ``kl_weight`` times KL(N(mu, sigma^2) || N(0, I)) summed over the code.
    Hidden layers use tanh; the output uses the TP3 logistic, sigmoid(2*h).
    """

    def __init__(self, encoder=(784, 256, 64), latent=2, kl_weight=1.0,
                 seed=42, optimizer=None, batch_size=64):
        self.encoder_widths = _widths(encoder, 'encoder')
        if type(latent) is not int or latent < 1:
            raise ValueError('latent must be a positive integer')
        if not np.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError('kl_weight must be finite and non-negative')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.latent = latent
        self.kl_weight = float(kl_weight)
        self.seed = int(seed)
        self.batch_size = batch_size
        self.decoder_widths = [latent] + self.encoder_widths[::-1][:-1] + [self.encoder_widths[0]]
        self._optimizer_config = dict(optimizer or {'name': 'adam', 'learning_rate': 0.001})
        self.optimizer = construir_optimizador(self._optimizer_config)
        self._tanh = construir_activacion('tanh', 1.0)
        self._logistic = construir_activacion('logistica', 1.0)
        init = np.random.default_rng(np.random.SeedSequence([self.seed, 0]))
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))

        def layer(n_in, n_out):
            # Same Xavier rule as tp3.shared.mlp: N(0, sqrt(2 / (n_in + n_out))).
            return init.normal(0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out)), np.zeros(n_out)

        self.enc = [layer(a, b) for a, b in zip(self.encoder_widths, self.encoder_widths[1:])]
        self.head_mu = layer(self.encoder_widths[-1], latent)
        self.head_logvar = layer(self.encoder_widths[-1], latent)
        self.dec = [layer(a, b) for a, b in zip(self.decoder_widths, self.decoder_widths[1:])]
        self.history = []

    # ----- parameters -------------------------------------------------
    @property
    def parameters(self):
        """Fixed order shared by gradients, optimizer and persistence."""
        layers = self.enc + [self.head_mu, self.head_logvar] + self.dec
        return [w for w, _ in layers] + [b for _, b in layers]

    def _input(self, values, width):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f'Expected finite nonempty (N,{width}) data')
        return values

    # ----- forward ----------------------------------------------------
    def _encode(self, x):
        acts = [x]
        for w, b in self.enc:
            acts.append(self._tanh.theta(acts[-1] @ w + b))
        mu = acts[-1] @ self.head_mu[0] + self.head_mu[1]
        logvar = acts[-1] @ self.head_logvar[0] + self.head_logvar[1]
        return acts, mu, logvar

    def _decode(self, z):
        acts = [z]
        for w, b in self.dec[:-1]:
            acts.append(self._tanh.theta(acts[-1] @ w + b))
        w, b = self.dec[-1]
        return acts, acts[-1] @ w + b  # pre-activation of the logistic output

    def encode(self, x):
        """Return (mu, logvar) of q(z|x)."""
        _, mu, logvar = self._encode(self._input(x, self.encoder_widths[0]))
        return mu, logvar

    def decode(self, z):
        _, pre = self._decode(self._input(z, self.latent))
        return self._logistic.theta(pre)

    def reconstruct(self, x):
        """Deterministic reconstruction through the mean code."""
        return self.decode(self.encode(x)[0])

    def sample(self, n, rng):
        """Decode n codes drawn from the prior N(0, I)."""
        z = rng.standard_normal((n, self.latent))
        return z, self.decode(z)

    # ----- loss and gradients -----------------------------------------
    @staticmethod
    def _bce_from_pre(x, pre):
        # Output is sigmoid(a) with a = 2*pre; BCE = softplus(a) - x*a, per sample.
        a = 2.0 * pre
        return (np.logaddexp(0.0, a) - x * a).sum(axis=1)

    @staticmethod
    def _kl(mu, logvar):
        return 0.5 * (np.exp(logvar) + mu ** 2 - 1.0 - logvar).sum(axis=1)

    def terms(self, x, eps=None):
        """Mean reconstruction BCE and mean KL; eps=None uses the mean code."""
        x = self._input(x, self.encoder_widths[0])
        _, mu, logvar = self._encode(x)
        z = mu if eps is None else mu + np.exp(0.5 * logvar) * eps
        _, pre = self._decode(z)
        return float(self._bce_from_pre(x, pre).mean()), float(self._kl(mu, logvar).mean())

    def gradients(self, x, eps):
        """Return (reconstruction, kl, gradients) for J = L + kl_weight * R."""
        n = x.shape[0]
        enc_acts, mu, logvar = self._encode(x)
        sigma = np.exp(0.5 * logvar)
        dec_acts, pre = self._decode(mu + sigma * eps)
        out = self._logistic.theta(pre)
        recon = float(self._bce_from_pre(x, pre).mean())
        kl = float(self._kl(mu, logvar).mean())

        # Decoder: dBCE/dpre = (o - x) / (o (1 - o)) * theta'(pre) = 2 (o - x).
        delta = 2.0 * (out - x) / n
        dec_w, dec_b = [], []
        for index in range(len(self.dec) - 1, -1, -1):
            w, _ = self.dec[index]
            dec_w.append(dec_acts[index].T @ delta)
            dec_b.append(delta.sum(axis=0))
            delta = delta @ w.T
            if index > 0:
                delta = delta * self._tanh.dtheta(None, dec_acts[index])
        dz = delta  # dL/dz, back through the stochastic node

        # Reparameterization: dz/dmu = 1, dz/dlogvar = 0.5 * sigma * eps.
        dmu = dz + self.kl_weight * mu / n
        dlogvar = dz * 0.5 * sigma * eps + self.kl_weight * 0.5 * (np.exp(logvar) - 1.0) / n
        h = enc_acts[-1]
        head_w = [h.T @ dmu, h.T @ dlogvar]
        head_b = [dmu.sum(axis=0), dlogvar.sum(axis=0)]
        delta = (dmu @ self.head_mu[0].T + dlogvar @ self.head_logvar[0].T) * self._tanh.dtheta(None, h)
        enc_w, enc_b = [], []
        for index in range(len(self.enc) - 1, -1, -1):
            w, _ = self.enc[index]
            enc_w.append(enc_acts[index].T @ delta)
            enc_b.append(delta.sum(axis=0))
            if index > 0:
                delta = (delta @ w.T) * self._tanh.dtheta(None, enc_acts[index])
        grads = enc_w[::-1] + head_w + dec_w[::-1] + enc_b[::-1] + head_b + dec_b[::-1]
        return recon, kl, grads

    # ----- training ---------------------------------------------------
    def train_epoch(self, x):
        """One pass over x in shuffled mini-batches; returns mean terms."""
        x = self._input(x, self.encoder_widths[0])
        order = self._rng.permutation(x.shape[0])
        recon_sum = kl_sum = 0.0
        for start in range(0, x.shape[0], self.batch_size):
            batch = x[order[start:start + self.batch_size]]
            eps = self._rng.standard_normal((batch.shape[0], self.latent))
            recon, kl, grads = self.gradients(batch, eps)
            self.optimizer.paso(self.parameters, grads)
            recon_sum += recon * batch.shape[0]
            kl_sum += kl * batch.shape[0]
        return recon_sum / x.shape[0], kl_sum / x.shape[0]

    # ----- persistence ------------------------------------------------
    @property
    def configuration(self):
        return {'encoder': self.encoder_widths, 'latent': self.latent,
                'decoder': self.decoder_widths, 'kl_weight': self.kl_weight,
                'seed': self.seed, 'batch_size': self.batch_size,
                'optimizer': self.optimizer.configuracion(),
                'hidden_activation': 'tanh', 'output': 'logistica',
                'reconstruction_loss': 'bce', 'initialization': 'xavier'}

    def save(self, path):
        """Weights and configuration for evaluation and generation.

        Optimizer moments and RNG state are not stored: a loaded model is
        meant to be evaluated, not to resume training identically.
        """
        arrays = {f'p{i}': p for i, p in enumerate(self.parameters)}
        np.savez(path, configuration=json.dumps(self.configuration),
                 history=json.dumps(self.history), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(data['configuration'].item())
            model = cls(encoder=config['encoder'], latent=config['latent'],
                        kl_weight=config['kl_weight'], seed=config['seed'],
                        optimizer=config['optimizer'], batch_size=config['batch_size'])
            for index, parameter in enumerate(model.parameters):
                stored = data[f'p{index}']
                if stored.shape != parameter.shape:
                    raise ValueError('Stored parameters do not match the configuration')
                np.copyto(parameter, stored)
            model.history = json.loads(data['history'].item())
        return model
