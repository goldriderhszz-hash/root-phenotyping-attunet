"""Root segmentation and mask-derived skeleton descriptors."""

__all__ = ["AttentionUNet", "extract_phenotypes_decoupled"]
__version__ = "1.0.0"


def __getattr__(name):
    """Load the PyTorch pipeline only when it is explicitly requested."""
    if name == "AttentionUNet":
        from .pipeline import AttentionUNet

        return AttentionUNet
    if name == "extract_phenotypes_decoupled":
        from .phenotypes import extract_phenotypes_decoupled

        return extract_phenotypes_decoupled
    raise AttributeError(name)
