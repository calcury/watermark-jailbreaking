"""NumPy geometry in the final normalized state space used by the LM head."""
import numpy as np


def fit_subspace(states, rank):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError('states must be a finite samples-by-features matrix')
    if rank < 1 or rank > min(states.shape[0] - 1, states.shape[1]):
        raise ValueError('rank exceeds the centered sample rank bound')
    center = states.mean(axis=0)
    _, singular, vt = np.linalg.svd(states - center, full_matrices=False)
    tolerance = np.finfo(float).eps * max(states.shape) * singular[0]
    numerical_rank = int(np.sum(singular > tolerance))
    if rank > numerical_rank:
        raise ValueError(f'requested rank {rank} exceeds numerical rank {numerical_rank}')
    energy = singular ** 2
    probabilities = energy[energy > 0] / energy.sum()
    return center, vt[:rank].T, {
        'samples': len(states), 'numerical_rank': numerical_rank,
        'singular_values': singular.tolist(),
        'participation_ratio': float(energy.sum() ** 2 / np.sum(energy ** 2)),
        'entropy_rank': float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        'explained_variance': float(energy[:rank].sum() / energy.sum()),
    }


def project(vector, basis):
    return basis @ (basis.T @ vector)


def state_metrics(state, center, basis):
    offset = state - center
    tangent = project(offset, basis)
    normal = offset - tangent
    energy = float(offset @ offset)
    return {
        'distance': float(np.linalg.norm(normal)),
        'normalized_distance': float(np.linalg.norm(normal) / max(np.sqrt(energy), 1e-12)),
        'projection_energy': float(tangent @ tangent / max(energy, 1e-24)),
        'center_distance': float(np.sqrt(energy)),
    }


def direction_metrics(english, translated, direction, bias, center, basis):
    """Exact linear margin decomposition, with no dense d-by-d projector."""
    delta = translated - english
    tangent = project(delta, basis)
    normal = delta - tangent
    retained_direction = project(direction, basis)
    direction_energy = float(direction @ direction)
    delta_energy = float(delta @ delta)
    margin_en = float(direction @ english + bias)
    margin_other = float(direction @ translated + bias)
    inside = float(direction @ tangent)
    outside = float(direction @ normal)
    projected_state = center + project(translated - center, basis)
    return {
        'margin_en': margin_en, 'margin_other': margin_other,
        'delta_margin': margin_other - margin_en,
        'inside_delta_margin': inside, 'outside_delta_margin': outside,
        'decomposition_error': float(abs((margin_other - margin_en) - inside - outside)),
        'direction_retention': float(retained_direction @ retained_direction / max(direction_energy, 1e-24)),
        'delta_norm': float(np.sqrt(delta_energy)),
        'outside_displacement': float(np.linalg.norm(normal)),
        'outside_fraction': float(normal @ normal / delta_energy) if delta_energy > 1e-24 else None,
        'distance_change': state_metrics(translated, center, basis)['distance'] - state_metrics(english, center, basis)['distance'],
        'projected_margin_other': float(direction @ projected_state + bias),
    }


def principal_angles(first, second):
    cosines = np.clip(np.linalg.svd(first.T @ second, compute_uv=False), 0, 1)
    return {'angles_degrees': np.degrees(np.arccos(cosines)).tolist(),
            'overlap': float(np.sum(cosines ** 2) / min(first.shape[1], second.shape[1]))}


def random_basis(dimension, rank, rng):
    return np.linalg.qr(rng.normal(size=(dimension, rank)), mode='reduced')[0]
