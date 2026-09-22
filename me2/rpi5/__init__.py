"""Pi-local runtime modules for the Pi5-VCM model and demos."""

__all__ = ["HarnessConfig", "PiHarness"]


def __getattr__(name):
	if name in __all__:
		from .harness import HarnessConfig, PiHarness
		return {"HarnessConfig": HarnessConfig, "PiHarness": PiHarness}[name]
	raise AttributeError(name)
