"""SCION-style tied Sign endpoint.

For this release, the SCION-style baseline is exposed as the tied-table
Sign endpoint of the released AF-Muon V2 optimizer family. This matches the paper
comparison arm: hidden matrices use Muon, vectors use RMS, and the tied
embedding/LM-head uses c=1, s=1.
"""

from .afmoun_v2 import build_scion_sign_v2 as build_scion_sign

__all__ = ["build_scion_sign"]
