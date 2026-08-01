"""Runtime configuration for the hidden MAGNET backend.

Developer note
-------------
Author: Nirmal Parmar, Machine Gnostics

This module keeps the backend selection logic out of the user-facing API.
MAGNET code should call ``configure`` once to choose the execution device and
numeric precision. The rest of the package reads the active runtime through
``get_runtime`` and converts values with ``to_torch`` / ``to_numpy``.

Bird's-eye view
---------------
- ``configure`` selects cpu, mac, cuda, or auto mode.
- ``configure`` also controls torch thread pools and notebook-safe OpenMP/MKL
	limits when requested.
- ``get_runtime`` returns the current device and dtype preferences.
- ``to_torch`` converts Python and NumPy values into backend tensors.
- ``to_numpy`` converts backend tensors back into plain NumPy arrays for the
  public MAGNET boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeConfig:
	"""Resolved MAGNET execution preferences.

	The object stores the user-facing runtime choices that MAGNET should honor
	for newly created tensors and training loops.

	Attributes
	----------
	device:
		Resolved execution target such as ``"cpu"``, ``"mps"``, or ``"cuda"``.
	dtype:
		Default floating-point precision expressed as a friendly MAGNET name.
	seed:
		Optional reproducibility seed applied to NumPy and torch.
	deterministic:
		Whether torch should try to use deterministic algorithms.
	"""

	device: str = "auto"
	dtype: str = "float32"
	seed: int | None = None
	deterministic: bool = False


_DTYPE_ALIASES = {
	"float16": torch.float16,
	"half": torch.float16,
	"float32": torch.float32,
	"fp32": torch.float32,
	"float64": torch.float64,
	"double": torch.float64,
	"bfloat16": torch.bfloat16,
}


_ACTIVE_RUNTIME = RuntimeConfig()


def _resolve_requested_device(device: str | None) -> str:
	requested = (device or _ACTIVE_RUNTIME.device).strip().lower()
	if requested in {"auto", "default", "best"}:
		if torch.cuda.is_available():
			return "cuda"
		if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
			return "mps"
		return "cpu"
	if requested == "cpu":
		return "cpu"
	if requested.startswith("cpu"):
		return "cpu"
	if requested in {"mac", "mps", "apple"}:
		if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
			return "mps"
		raise RuntimeError("MAGNET device 'mac' requires Apple Metal Performance Shaders support")
	if requested.startswith("mps"):
		if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
			return requested
		raise RuntimeError("MAGNET device 'mps' requires Apple Metal Performance Shaders support")
	if requested in {"cuda", "gpu", "nvidia"} or requested.startswith("cuda:"):
		if torch.cuda.is_available():
			return requested if requested.startswith("cuda:") else "cuda"
		raise RuntimeError("MAGNET device 'cuda' requires CUDA support")
	raise ValueError(f"Unsupported MAGNET device: {device!r}")


def _resolve_dtype(dtype: str | None):
	requested = (dtype or _ACTIVE_RUNTIME.dtype).strip().lower()
	try:
		return _DTYPE_ALIASES[requested]
	except KeyError as exc:
		raise ValueError(f"Unsupported MAGNET dtype: {dtype!r}") from exc


def friendly_device_name(device: str | torch.device) -> str:
	"""Return a user-friendly MAGNET device name.

	This keeps torch-style suffixes such as ``":0"`` out of the public-facing
	tensor representation while preserving the actual backend selection in the
	internal runtime.
	"""
	device_text = str(device)
	if device_text.startswith("cuda"):
		return "cuda"
	if device_text.startswith("mps"):
		return "mps"
	if device_text.startswith("cpu"):
		return "cpu"
	return device_text


def configure(
	device: str = "auto",
	dtype: str = "float32",
	seed: int | None = 42,
	deterministic: bool = False,
	mps_fallback: bool = True,
	threads: int | None = None,
	interop_threads: int | None = None,
	omp_threads: int | None = None,
	mkl_threads: int | None = None,
) -> RuntimeConfig:
	"""Configure the hidden MAGNET backend.

	Use this once near the start of your program to choose where MAGNET runs.
	The backend stays hidden, but the device choice is still explicit so users
	can switch between CPU, Apple silicon, and CUDA training without touching
	any torch API.

	Parameters
	----------
	device:
		Requested execution target. ``"auto"`` picks the best available device.
	dtype:
		Default precision for newly created tensors.
	seed:
		Optional reproducibility seed.
	deterministic:
		Ask torch for deterministic algorithms when available.
	mps_fallback:
			If ``True`` and MPS is selected, enable PyTorch's CPU fallback for
			unsupported MPS operations.
	threads:
		Optional torch intra-op thread count to apply after runtime setup.
	interop_threads:
			Optional torch inter-op thread count to apply after runtime setup.
	omp_threads:
		Optional OpenMP thread limit. If omitted on Apple Silicon/MPS, MAGNET
		defaults this to ``1`` to reduce notebook crash risk.
	mkl_threads:
		Optional MKL thread limit. If omitted on Apple Silicon/MPS, MAGNET
		defaults this to ``1`` to reduce notebook crash risk.

	Returns
	-------
	RuntimeConfig
		The active runtime configuration after resolution.
	"""
	global _ACTIVE_RUNTIME
	resolved_device = _resolve_requested_device(device)
	resolved_dtype = dtype.lower()
	_ACTIVE_RUNTIME = replace(RuntimeConfig(device=resolved_device, dtype=resolved_dtype, seed=seed, deterministic=deterministic))
	if seed is not None:
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	if mps_fallback and resolved_device.startswith("mps"):
		os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
		if omp_threads is None:
			omp_threads = 1
		if mkl_threads is None:
			mkl_threads = 1
	if omp_threads is not None:
		os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))
	if mkl_threads is not None:
		os.environ["MKL_NUM_THREADS"] = str(int(mkl_threads))
	if deterministic:
		torch.use_deterministic_algorithms(True, warn_only=True)
	if threads is not None:
		torch.set_num_threads(int(threads))
	if interop_threads is not None:
		torch.set_num_interop_threads(int(interop_threads))
	return _ACTIVE_RUNTIME


def get_runtime() -> RuntimeConfig:
	"""Return the current MAGNET runtime configuration.

	This is useful for debugging or for showing the resolved device selection in
	application logs without exposing any torch-specific objects to the user.
	"""
	return _ACTIVE_RUNTIME


def get_torch_device(device: str | None = None) -> torch.device:
	"""Resolve a MAGNET device string into a torch device.

	This helper is internal to MAGNET. It exists so the backend can translate
	friendly device names such as ``"auto"`` or ``"mac"`` into the actual torch
	device object required by the implementation.
	"""
	return torch.device(_resolve_requested_device(device))


def get_torch_dtype(dtype: str | None = None):
	"""Resolve a MAGNET dtype string into a torch dtype."""
	return _resolve_dtype(dtype)


def to_torch(value, *, requires_grad: bool = False, device: str | None = None, dtype: str | None = None) -> torch.Tensor:
	"""Convert a MAGNET value to a torch tensor without exposing torch upstream.

	The returned tensor is suitable for internal MAGNET use only. User code
	continues to interact with the lightweight ``Tensor`` facade.
	"""
	if hasattr(value, "_tensor") and isinstance(getattr(value, "_tensor"), torch.Tensor):
		tensor = value._tensor.detach().clone()
		if device is not None or dtype is not None:
			tensor = tensor.to(device=get_torch_device(device), dtype=get_torch_dtype(dtype))
	else:
		tensor = torch.as_tensor(value, device=get_torch_device(device), dtype=get_torch_dtype(dtype))
	if requires_grad:
		tensor = tensor.detach().clone().requires_grad_(True)
	else:
		tensor = tensor.detach()
	return tensor


def to_numpy(value) -> np.ndarray:
	"""Convert a tensor-like value into a detached NumPy array.

	This is the boundary conversion used whenever MAGNET needs to present a
	result in the user-facing NumPy style that the current API expects.
	"""
	if hasattr(value, "_tensor") and isinstance(getattr(value, "_tensor"), torch.Tensor):
		value = value._tensor
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy()
	return np.asarray(value)
