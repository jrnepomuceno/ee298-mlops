"""Fixed offline reply text for the VCM intent catalog."""

INTENT_REPLIES = {
    "turn_on_lights": "The lights are on.",
    "turn_off_lights": "The lights are off.",
    "dim_lights": "The lights have been dimmed.",
    "set_temperature": "The temperature has been set.",
    "play_music": "Playing music.",
    "pause_music": "Music paused.",
    "stop_music": "Music stopped.",
    "set_timer": "Timer set.",
    "set_alarm": "Alarm set.",
    "cancel_timer": "Timer cancelled.",
    "remind": "Reminder saved.",
    "call": "Starting the call.",
    "what_time": "The current time is available.",
    "what_weather": "The weather is available.",
    "what_reminders": "Here are your reminders.",
    "oov": "I did not recognize that command.",
}

WAKE_REPLY = "Yes?"
