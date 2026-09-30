from .afmoun_v2 import AFMuonV2, build_afmoun_v2, build_scion_sign_v2
from .muon import HybridMuon, build_hybrid_muon

AFMuon = AFMuonV2
build_afmoun = build_afmoun_v2
build_scion_sign = build_scion_sign_v2

__all__ = [
    "AFMuon",
    "AFMuonV2",
    "HybridMuon",
    "build_afmoun",
    "build_afmoun_v2",
    "build_hybrid_muon",
    "build_scion_sign",
    "build_scion_sign_v2",
]
