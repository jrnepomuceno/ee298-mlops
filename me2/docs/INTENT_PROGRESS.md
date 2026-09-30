# RPi5 Intent Progress

**Last updated:** 2026-09-30
**Recognition backend:** deterministic mock; no model-based recognition has been evaluated here.

This matrix tracks task/harness behavior independently of model accuracy. “Mock passed” means a known intent and representative slots reached the facade/orchestrator and produced the expected dry-run task/reply. “Live checked” means the corresponding Pi peripheral or service was actually exercised with recognition still mocked.

| Intent | Mock path | Live Pi path | Notes |
|---|---|---|---|
| `turn_on_lights` | Passed | Passed | DuoCast on command; cleanup restores off. |
| `turn_off_lights` | Passed | Passed | DuoCast off command. |
| `dim_lights` | Passed | Passed | Tested at 40%; cleanup restores off. |
| `set_temperature` | Passed | Mic-to-Piper passed | Latest retry captured 1.23 s from HyperX DuoCast; forced mock 25 C and Piper played “Temperature set to 25 degrees” with 750 ms silence pre-roll; user confirmed the first word was audible. No thermostat control; `side_effects=false`. |
| `play_music` | Passed | Not tested | Requires a real music directory/player selection. |
| `pause_music` | Passed | Not tested | Live media playback not enabled. |
| `stop_music` | Passed | Not tested | Live media playback not enabled. |
| `set_timer` | Passed | Passed | Pi TimerManager expired a 3 s countdown; persistent TimerAlarm looped the chime and announced “Your timer is up.”; mock `stop_timer` stopped it after the test ring window. |
| `set_alarm` | Passed | Not tested | Alarm scheduling/expiry playback not exercised live. |
| `stop_timer` | Passed | Passed | Mock stop intent stopped an expired ringing alarm; separate pre-expiry cancel test verified no later expiry/ringtone. |
| `remind` | Passed | Not tested | Persistent reminder write/read not exercised in this pass. |
| `call` | Passed | Not tested | No SIP call placed; requires a test target and explicit confirmation. |
| `what_time` | Passed | Passed | Piper synthesized and played the reply through `pw-play` to the Pi's default speaker. |
| `what_weather` | Passed | Passed | One OpenWeatherMap request succeeded for Quezon City: 33 C, cloudy. |
| `what_reminders` | Passed | Not tested | Live reminder-store query not exercised in this pass. |
| `volume_up` | Passed | Passed | Pi master volume 58% → 68%; snapshot restored to 58%. |
| `volume_down` | Passed | Passed | Pi master volume 58% → 48%; snapshot restored to 58%. |
| `mute` | Disabled | Disabled | ONNX label retained for index compatibility; facade, mock catalog, and harness reject it. |
| `oov` | Passed | Rejection | OOV/unknown intents now reply “I don't understand.” |

## Shared Rejection Cases

| Condition | Expected reply |
|---|---|
| Missing required slot | “I don't understand.” |
| Low confidence | “I don't understand.” |
| OOV or unknown intent | “I don't understand.” |
| Invalid or out-of-range slot value | “Invalid value. Try again.” |

## Test Evidence

The Pi-side focused suites passed during this test cycle: facade 34 tests, HVAC 11, mock intents 14, lights 21, playback/pre-roll 6, and timer/TimerAlarm 30. Live timer start/expiry and cancellation were exercised with mocked recognition; production ringing continues until the model returns `stop_timer`. This validates timer/task/audio wiring, not model recognition quality.

The DuoCast uses a persistent `quadcastrgb` process. The driver now stops its prior process before changing state and uses the installed CLI's documented `solid 0` off command. The ring was confirmed left off after testing. The wireless speaker appears to need a longer active stream before speech; 50 ms was insufficient, while the user confirmed the initial word was audible with 750 ms digital-silence pre-roll. The weather key must be exported to the Python process; it was exported only for the one successful request and was not printed.

The mock runner's `--live-volume` mode snapshots the Pi master level before one `volume_up` or `volume_down` intent, records the step, and restores the snapshot in cleanup. Both commands were tested sequentially from 58%; the final `wpctl` readback was 58%.

## Updating This Matrix

After each intent, executor, peripheral, or response revision, update the relevant row and the date. Mark mock and live results separately. Record the exact slots/physical path tested and any cleanup state. Never mark a live path passed based only on a dry-run or mock result; keep model accuracy and recognition metrics in the model validation report.