"""One function that calls an LLM. Two styles are supported, chosen with LLM_PROVIDER.

1) OpenAI-compatible endpoints (the default; hosted providers or a local Ollama server):
    $env:LLM_BASE_URL = "..."   # the provider's OpenAI-compatible base URL
    $env:LLM_MODEL    = "..."   # a model name from the provider's docs
    $env:LLM_API_KEY  = "..."   # not needed for a local Ollama server

2) Google's native Gemini API (use this with the newer "AQ." keys from AI Studio):
    $env:LLM_PROVIDER = "gemini"
    $env:LLM_MODEL    = "..."   # a Gemini model name, for example from AI Studio's code sample
    $env:LLM_API_KEY  = "..."

If the needed variables are missing, generate() raises LLMNotConfigured.
Stray spaces around the values are trimmed, and the API key is hidden in error messages.

Optional: if calls take minutes because your network's IPv6 route is broken (Python waits for
IPv6 to time out before trying IPv4), force IPv4 with:
    $env:LLM_FORCE_IPV4 = "1"
"""
import os
import socket
import time

import requests

GEMINI_NATIVE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Temporary problems worth retrying: rate limits (429) and server-side overload or errors.
# Anything else (a bad key, a wrong model name) fails immediately, because retrying cannot help.
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMNotConfigured(Exception):
    """No LLM has been set up through environment variables."""


class LLMError(Exception):
    """The LLM provider returned an error or an unexpected reply."""


def _env(name):
    """Read an environment variable and trim stray whitespace (a pasted key often has some)."""
    return (os.getenv(name) or "").strip()


def _redact(text):
    """Never let the API key appear in an error message."""
    key = _env("LLM_API_KEY")
    return text.replace(key, "***") if key else text


def _maybe_force_ipv4():
    """Opt-in fix for networks where IPv6 is advertised but does not work.

    requests (through urllib3) tries a host's IPv6 addresses first and waits for each to time
    out before falling back to IPv4. Telling urllib3 to ask only for IPv4 addresses skips that wait.
    """
    if _env("LLM_FORCE_IPV4") == "1":
        import urllib3.util.connection as urllib3_connection
        urllib3_connection.allowed_gai_family = lambda: socket.AF_INET


def _parse_openai(data):
    return data["choices"][0]["message"]["content"].strip()


def _parse_gemini(data):
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts).strip()


def _build_request(prompt):
    """Return (url, headers, json_body, parser) for the configured provider."""
    provider = (_env("LLM_PROVIDER") or "openai").lower()
    model = _env("LLM_MODEL")
    api_key = _env("LLM_API_KEY")

    if provider == "gemini":
        if not model or not api_key:
            raise LLMNotConfigured("With LLM_PROVIDER=gemini, set LLM_MODEL and LLM_API_KEY.")
        model = model.removeprefix("models/")
        return (
            f"{GEMINI_NATIVE_URL}/models/{model}:generateContent",
            {"x-goog-api-key": api_key},
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": {"temperature": 0}},
            _parse_gemini,
        )

    base_url = _env("LLM_BASE_URL")
    if not base_url or not model:
        raise LLMNotConfigured("Set LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY if required).")
    return (
        base_url.rstrip("/") + "/chat/completions",
        {"Authorization": f"Bearer {api_key or 'not-needed'}"},
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        _parse_openai,
    )


def generate(prompt, timeout=30, max_attempts=3):
    _maybe_force_ipv4()
    url, headers, body, parse = _build_request(prompt)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as error:
            last_error = LLMError(_redact(f"Could not reach the LLM: {error}"))
        else:
            if response.status_code == 200:
                try:
                    return parse(response.json())
                except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
                    raise LLMError("Unexpected reply format from the LLM.") from error
            last_error = LLMError(_redact(f"LLM request failed ({response.status_code}): {response.text[:200]}"))
            if response.status_code not in RETRY_STATUS:
                raise last_error
        if attempt < max_attempts:
            time.sleep(2 ** attempt)  # wait 2 seconds, then 4 seconds
    raise last_error