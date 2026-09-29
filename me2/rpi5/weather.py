"""OpenWeatherMap weather provider for the ``what_weather`` intent.

Turns a live current-weather lookup into a single spoken-friendly line, e.g.
``"It's 24 degrees and partly cloudy in Quezon City."``

The API key is read from the environment (``OPENWEATHER_API_KEY``) at call
time -- it is never written to the repo. Location defaults to Quezon City and
can be overridden with ``WEATHER_LOCATION``.

Only the standard library is used (urllib), so this stays lightweight on the
RPi5 and testable without third-party HTTP clients.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Callable

LOGGER = logging.getLogger("pi5-vcm.weather")

DEFAULT_LOCATION = "Quezon City"
API_BASE = "https://api.openweathermap.org/data/2.5/weather"
TIMEOUT_SECONDS = 8.0

# Map OpenWeatherMap ``weather[].main`` codes to words that read naturally
# when spoken. Anything unmapped falls back to the raw ``description``.
_CONDITION_WORDS = {
    "Clear": "clear skies",
    "Clouds": "cloudy",
    "Rain": "rainy",
    "Drizzle": "drizzly",
    "Thunderstorm": "thunderstorms",
    "Mist": "misty",
    "Fog": "foggy",
    "Smoke": "smoky",
    "Haze": "hazy",
    "Dust": "dusty",
    "Ash": "ashy",
    "Squall": "squally",
    "Tornado": "tornadic",
}


def _condition_word(main: str, description: str) -> str:
    """Return a spoken condition phrase from OWM ``main``/``description``."""
    word = _CONDITION_WORDS.get((main or "").strip())
    if word:
        return word
    # Fall back to the human description, lower-cased and stripped of
    # trailing punctuation so it reads cleanly in a sentence.
    desc = (description or "unknown conditions").strip().rstrip(".")
    return desc.lower() if desc else "unknown conditions"


def _fetch_current(location: str, api_key: str) -> dict:
    """Call OpenWeatherMap current-weather and return the parsed JSON body."""
    params = urllib.parse.urlencode({
        "q": location,
        "appid": api_key,
        "units": "metric",
    })
    url = f"{API_BASE}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "pi5-vcm/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def make_weather_fn(location: str | None = None,
                    api_key: str | None = None,
                    timeout: float = TIMEOUT_SECONDS) -> Callable[[], str]:
    """Build a zero-arg ``() -> str`` provider for :class:`InfoExecutor`.

    Args:
        location: City to query. Defaults to ``WEATHER_LOCATION`` env var,
            then :data:`DEFAULT_LOCATION`.
        api_key: OpenWeatherMap key. Defaults to ``OPENWEATHER_API_KEY`` env var.
        timeout: Per-request socket timeout in seconds.

    Returns:
        A callable that performs one live lookup and returns a spoken line.
        Raises :class:`RuntimeError` if no API key is configured or the lookup
        fails -- the executor catches that and speaks a graceful apology.
    """
    loc = (location or os.environ.get("WEATHER_LOCATION") or DEFAULT_LOCATION).strip()

    def _weather() -> str:
        key = (api_key or os.environ.get("OPENWEATHER_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENWEATHER_API_KEY is not set")
        data = _fetch_current(loc, key)
        if data.get("cod") not in (200, "200"):
            message = data.get("message", "unknown error")
            raise RuntimeError(f"weather service error: {message}")
        temp_c = int(round(float(data["main"]["temp"])))
        weather = (data.get("weather") or [{}])[0]
        condition = _condition_word(weather.get("main", ""), weather.get("description", ""))
        return f"It's {temp_c} degrees and {condition} in {loc}."

    _weather.__name__ = "weather"
    return _weather
