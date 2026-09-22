"""Pi-resident MCP adapter for the mini-Alexa runtime.

The server is intentionally thin: all recognition, gating, dry-run action
handling, and reply generation remain in PiHarness.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .audio import microphone_utterances
from .harness import HarnessConfig, PiHarness
from .rgb import RgbController
from .replies import reply_wav_name
from .tts import WavPlayer
from .wakeword import OpenWakeWordDetector


def create_server(harness: PiHarness, wakeword: Any | None = None,
                  rgb: RgbController | None = None,
                  wav_player: WavPlayer | None = None,
                  reply_dir: str = "assets/replies"):
    """Create an MCP server whose tools delegate to one loaded harness."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP mode requires the mcp package; install requirements-mcp.txt"
        ) from exc

    server = FastMCP("pi5-vcm")

    @server.tool()
    def health() -> dict[str, Any]:
        """Report local model readiness and runtime identity."""
        return {
            "status": "ready",
            "device": str(harness.device),
            "checkpoint": str(harness.checkpoint),
            "side_effects": False,
        }

    @server.tool()
    def recognize_file(path: str) -> dict[str, Any]:
        """Recognize one local 16 kHz WAV and return the assistant event."""
        return harness.recognize_wav(path)

    @server.tool()
    def listen_once() -> dict[str, Any]:
        """Listen for one VAD-bounded utterance and return its event."""
        controller = rgb or RgbController()
        wav = next(microphone_utterances(wakeword=wakeword,
                                         on_wake=controller.wake))
        controller.speaking()
        event = harness.recognize_audio(wav, "microphone")
        if event["reply"].get("speak"):
            wav_name = reply_wav_name(event["result"], event["action"])
            event["tts"] = (wav_player or WavPlayer()).play(
                str(Path(reply_dir) / wav_name))
        event["rgb"] = controller.off()
        return event

    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi5-VCM MCP server")
    parser.add_argument("--checkpoint", default="inference/best.pt")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--wakeword", metavar="NAME",
                        help="optional pretrained openWakeWord model, e.g. alexa")
    parser.add_argument("--wakeword-threshold", type=float, default=0.7)
    parser.add_argument("--rgb-executable", metavar="PATH",
                        help="optional quadcastrgb executable for DuoCast LEDs")
    parser.add_argument("--audio-player", default="pw-play")
    parser.add_argument("--audio-device",
                        help="ALSA device passed to aplay, e.g. plughw:2,0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness = PiHarness(HarnessConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
    ))
    detector = (OpenWakeWordDetector(args.wakeword, args.wakeword_threshold)
                if args.wakeword else None)
    server = create_server(
        harness, detector, RgbController(args.rgb_executable),
        WavPlayer(args.audio_player, args.audio_device))
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
