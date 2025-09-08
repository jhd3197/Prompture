import requests
from ..core import Driver
from typing import Any, Dict

class LocalHTTPDriver(Driver):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def generate(self, prompt: str, options: Dict[str,Any]) -> Dict[str,Any]:
        payload = {"prompt": prompt, "options": options}
        r = requests.post(self.endpoint, json=payload, timeout=30)
        r.raise_for_status()
        # se espera {"text": "...", "meta": {...}}
        return r.json()