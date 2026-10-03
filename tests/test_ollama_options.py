"""Ollama reads sampling and length controls from `options`, not the top level."""

from prompture.drivers import ollama_driver
from prompture.drivers._ollama_options import apply_sampling


def test_controls_go_where_ollama_reads_them():
    payload = apply_sampling({"model": "m"}, {"temperature": 0.2, "top_p": 0.9,
                                              "max_tokens": 400, "think": False})
    assert payload == {"model": "m", "think": False,
                       "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 400}}
    assert apply_sampling({}, {"timeout": 30}) == {}


def test_a_chat_request_carries_the_reply_limit(monkeypatch):
    sent = {}

    class Reply:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}

    def post(url, json=None, timeout=None):
        sent.update(json)
        return Reply()

    monkeypatch.setattr(ollama_driver.requests, "post", post)
    driver = ollama_driver.OllamaDriver(endpoint="http://localhost:11434/api/generate",
                                        model="m")
    driver.generate_messages([{"role": "user", "content": "hi"}],
                             {"max_tokens": 50, "temperature": 0})
    assert sent["options"] == {"num_predict": 50, "temperature": 0}
    assert "temperature" not in sent and "max_tokens" not in sent
