"""
Factory for creating model launchers based on backend type.
"""
from supervisor.launchers.base import ModelLauncher
from supervisor.launchers.vllm_launcher import VLLMLauncher
from supervisor.launchers.sglang_launcher import SGLangLauncher
from supervisor.launchers.flux_launcher import FluxLauncher
from supervisor.launchers.clip_launcher import CLIPLauncher
from supervisor.launchers.species_launcher import SpeciesLauncher
from supervisor.launchers.face_launcher import FaceLauncher
from supervisor.launchers.dspark_launcher import DsparkLauncher
from supervisor.models import Backend


class LauncherFactory:
    """Factory for creating model launchers."""

    def __init__(self):
        self._launchers = {
            Backend.VLLM: VLLMLauncher(),
            Backend.SGLANG: SGLangLauncher(),
            Backend.FLUX: FluxLauncher(),
            Backend.CLIP: CLIPLauncher(),
            Backend.SPECIES: SpeciesLauncher(),
            Backend.FACE: FaceLauncher(),
            Backend.DSPARK: DsparkLauncher(),
            # Backend.TRT_LLM: TRTLLMLauncher(),  # TODO: Implement
        }

    def get_launcher(self, backend: Backend) -> ModelLauncher:
        """
        Get launcher for specified backend.

        Args:
            backend: Backend type

        Returns:
            Model launcher instance

        Raises:
            ValueError: If backend not supported
        """
        launcher = self._launchers.get(backend)
        if launcher is None:
            raise ValueError(f"Unsupported backend: {backend}")
        return launcher
