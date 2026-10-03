"""Model registry helpers for MAGNET."""

from __future__ import annotations

from .neuron import GnosticNeuron
from .sequential import Sequential


def get_model(model_name: str):
	"""Return a model class by registry name.

	Parameters
	----------
	model_name:
		Model registry key such as ``"sequential"`` or ``"gnostic_neuron"``.

	Returns
	-------
	type | None
		Resolved model class, or ``None`` when no match exists.
	"""
	key = model_name.replace("-", "_").lower()
	registry = {
		"sequential": Sequential,
		"gnostic_neuron": GnosticNeuron,
		"gnosticneuron": GnosticNeuron,
	}
	return registry.get(key)
