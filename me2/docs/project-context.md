ME2 - Voice Controlled Smart Device 

ASR models are not desirable for on-device computing because of footprint. The goal of this ME is to build a tiny Voice Command Model (VCM) that can understand the most common commands humans tell their smart devices:

# Most common commands issued to smart devices, enumerated

Ranked by real-world log and survey data (259,164 logged commands from Amazon Alexa + Google Home devices, plus recurring U.S. consumer surveys):

## Top commands, by rank

1. **Play music** — the #1 use case in every survey year (2018–2020). Actual forms: "play music"
2. **Ask a question / search** — *"what's the weather"* (daily-use #2–3), *"what time is it"*, 
3. **Control lights (IoT)** — 85% of all Alexa IoT commands: *"turn on/off lights"*,
4. **Dim / color lights** — ~10% of IoT commands: *"Dim lights to X percent"*
5. **Set a timer** — *"set a timer for X minutes"* 
6. **Set an alarm** — *"set an alarm for X am/pm"*
7. **Adjust thermostat temperature** — *"set temperature to X degrees"*
8. **Media control** — *"pause"*, *"stop"*, *"next/skip"*, *"volume up/down"*, *"louder"*. No. 1 can be folded to No. 8.
9. **Reminders and lists** — *"remind me to…"* "what are my reminders"
10. **Calls and messaging** — *"call mom"*

Your tasks:
1. Build a dataset to train VCMs - collective work
2.  Build and train a VCM on this dataset - individual work
3. Design a benchmark for validating VCMs - collective work
4. Validate the performance of your VCM  - individual work
5. Build a real-world demo of your VCM (this can run on RPi4/5 device) - individual work (may share devices)
6. Your VCM must be tiny - can run on RPi4 or 5 in real-time - individual work
7. VCM should be standalone - can not call cloud-based models.
8. No LLM, just pure VCM doing 1 to 10. Everything on-device.