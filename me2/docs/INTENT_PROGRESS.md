# RPi5 Intent Progress

**Last updated:** 2026-10-01
**Recognition backend:** deterministic mock; no model-based recognition has been evaluated here.

This matrix tracks task/harness behavior independently of model accuracy. “Mock passed” means a known intent and representative slots reached the facade/orchestrator and produced the expected dry-run task/reply. “Live checked” means the corresponding Pi peripheral or service was actually exercised with recognition still mocked.

| Intent | Mock path | Live Pi path | Notes |
|---|---|---|---|
| `turn_on_lights` | Passed | Passed | DuoCast warm-white command remains applied after assistant lifecycle returns to idle; visual confirmation pending. |
| `turn_off_lights` | Passed | Passed | DuoCast off command. |
| `dim_lights` | Passed | Passed | Tested at 40%; dimmed color remains applied after assistant lifecycle returns to idle. |
| `set_temperature` | Passed | Mic-to-Piper passed | Forced 25 C; Piper spoke “Temperature set to 25 degrees.” with 750 ms silence pre-roll; user confirmed the first word was audible. No thermostat control; `side_effects=false`. |
| `play_music` | Passed | Passed | Piper announces the title before playback. After pause, `play_music` resumes the same `ffplay` process and track and announces “Resuming <title>.” |
| `pause_music` | Passed | Passed | Paused active `ffplay` playback; Piper spoke “Music paused.” Static fallback WAV was rendered with the Pi Piper voice. |
| `stop_music` | Passed | Passed | Stopped active `ffplay` playback; Piper spoke “Music stopped.” Static fallback WAV was rendered with the Pi Piper voice. |
| `set_timer` | Passed | Passed | Piper says “Setting timer for <duration>. Starting now.” before countdown scheduling. A 5-second countdown expired, looped the chime, and spoke “Your timer is up.” |
| `set_alarm` | Passed | Schedule/cancel passed | Scheduled an alarm five minutes ahead, spoke the confirmation, then canceled it before expiry. Actual wall-clock expiry/audio not tested. |
| `stop_timer` | Passed | Passed | Stopped expired ringing. Countdown expiry delivery is serialized with cancellation; pre-expiry cancellation produced no later expiry or chime. |
| `remind` | Passed | Passed | Three reminders were persisted to an isolated temporary Pi JSON store, reloaded, and confirmed. The main reminder store was untouched. |
| `call` | Passed | Not tested | No SIP call placed; requires a test target and explicit confirmation. |
| `what_time` | Passed | Passed | Piper synthesized and played the reply through `pw-play` to the Pi's default speaker. |
| `what_weather` | Passed | Passed | Live OpenWeatherMap request for Quezon City returned 32 C, rainy; Piper reply playback passed. API key came from the Pi config and was not printed. |
| `what_reminders` | Passed | Passed | Three reminders were reloaded and spoken as an intro plus three item clips with 500 ms gaps. Existing user reminders were untouched. |
| `volume_up` | Passed | Passed | Pi master volume 73% → 83%; Piper reply played; snapshot restored to 73%. |
| `volume_down` | Passed | Passed | Pi master volume 73% → 63%; Piper reply played; snapshot restored to 73%. |
| `mute` | Disabled | Disabled | ONNX label retained for index compatibility; facade, mock catalog, and harness reject it. |
| `oov` | Passed | Rejection | OOV/unknown intents now reply “I don't understand.” |

The newer training package contains a `NEXT` label, but it is not part of the deployed model or the current intent catalog. The training-only model is not compatible with the Pi's deployed ONNX contract; there is no production `next_music` intent yet.

## Shared Rejection Cases

| Condition | Expected reply |
|---|---|
| Missing required slot | “I don't understand.” |
| Low confidence | “I don't understand.” |
| OOV or unknown intent | “I don't understand.” |
| Invalid or out-of-range slot value | “Invalid value. Try again.” |

## Test Evidence

Latest focused Pi test runs passed: timers 34, media 29, mock intents 15, reminders 31, and TTS 7. The light suite passes 25 tests locally; Pi on/dim/off actions were exercised through the production orchestrator. These runs validate task, executor, audio, and peripheral wiring with forced/mock intents, not model recognition accuracy.

The timer expiry cue uses the DuoCast's native red `pulse` mode. The Pi reported the pulse command applied and cleared the ring on `stop_timer`, but the user has not confirmed seeing the visual cue. `quadcastrgb --help` omits `pulse`; the installed man page and native implementation confirm it is supported. Light on/dim state now survives assistant idle/speaking transitions; visual confirmation remains pending. The controller stops its previous process and uses `solid 0` to turn the ring off.

The Pi Piper voice is used for the `dim_lights`, `set_timer`, `pause_music`, and `stop_music` fallback WAVs. `dim_lights.wav` says “Dimming lights.” with a per-intent slower setting; the WAV was generated on the Pi with Piper and verified byte-identical after pull. The pause/stop WAVs were also rendered on the Pi, and the reply generator skips these Piper-only assets when run on a host that only has espeak-ng.

The mock runner's `--live-volume` mode snapshots the Pi master level before one `volume_up` or `volume_down` intent, records the step, and restores the snapshot in cleanup. Both commands were tested sequentially from 73%; final `wpctl` readback was 73%.

## Updating This Matrix

After each intent, executor, peripheral, or response revision, update the relevant row and the date. Mark mock and live results separately. Record the exact slots/physical path tested and any cleanup state. Never mark a live path passed based only on a dry-run or mock result; keep model accuracy and recognition metrics in the model validation report.