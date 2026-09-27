"""model package: VCM architecture + training code.

Re-exports the VCM class and helpers so existing "from model import VCM"
keeps working after the move into the model/ subfolder.
"""
from model.model import VCM, ctc_decode_batch, parse_slots

__all__ = ["VCM", "ctc_decode_batch", "parse_slots"]
