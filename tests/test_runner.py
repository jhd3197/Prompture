from prompture import run_suite_from_spec
from prompture.drivers import MockDriver


def test_run_suite_from_spec():
    # Test spec with mock model and simple user info extraction test
    spec = {
        "meta": {"project": "test"},
        "models": [{"id": "mock1", "driver": "mock", "options": {}}],
        "tests": [
            {
                "id": "t1",
                "prompt_template": "Extract user info: '{text}'",
                "inputs": [{"text": "Juan is 28 and lives in Miami. He likes basketball and coding."}],
                "schema": {"type": "object", "required": ["name", "interests"]}
            }
        ]
    }
    
    # Setup drivers with mock driver
    drivers = {"mock": MockDriver()}
    
    # Run the test suite
    report = run_suite_from_spec(spec, drivers)
    
    # Verify report structure and content
    assert report["meta"]["project"] == "test"
    assert len(report["results"]) == 1
    
    result = report["results"][0]
    assert result["test_id"] == "t1"
    assert result["model_id"] == "mock1"
    assert result["validation"]["ok"] is True
    assert "name" in result["response"]
    assert "interests" in result["response"]
    assert result["response"]["name"] == "Juan"
    assert isinstance(result["response"]["interests"], list)
    assert len(result["response"]["interests"]) > 0