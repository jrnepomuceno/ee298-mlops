"""Pi-local runtime modules for the Pi5-VCM model and demos."""

__all__ = [
    "HarnessConfig", "PiHarness", "FacadePipeline",
    "ActionRequest", "RejectResult", "decode",
    "Orchestrator", "OrchestratorResult", "default_orchestrator",
]


def __getattr__(name):
    if name in ("HarnessConfig", "PiHarness", "FacadePipeline"):
        from .harness import HarnessConfig, PiHarness, FacadePipeline
        return {"HarnessConfig": HarnessConfig,
                "PiHarness": PiHarness,
                "FacadePipeline": FacadePipeline}[name]
    if name in ("ActionRequest", "RejectResult", "decode"):
        from .facade import ActionRequest, RejectResult, decode
        return {"ActionRequest": ActionRequest,
                "RejectResult": RejectResult,
                "decode": decode}[name]
    if name in ("Orchestrator", "OrchestratorResult", "default_orchestrator"):
        from .orchestrator import Orchestrator, OrchestratorResult, default_orchestrator
        return {"Orchestrator": Orchestrator,
                "OrchestratorResult": OrchestratorResult,
                "default_orchestrator": default_orchestrator}[name]
    raise AttributeError(name)
