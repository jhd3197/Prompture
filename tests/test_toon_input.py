"""Tests for TOON input conversion functions."""

import json
from unittest.mock import Mock, patch

import pytest

from prompture.core import (
    _calculate_token_savings,
    _dataframe_to_toon,
    _json_to_toon,
    extract_from_data,
    extract_from_pandas,
)


class TestJsonToToon:
    """Test _json_to_toon helper function."""

    def test_json_to_toon_list_input(self):
        """Test converting list of dicts to TOON."""
        data = [{"id": 1, "name": "Product A", "price": 10.0}, {"id": 2, "name": "Product B", "price": 20.0}]

        with patch("prompture.core.toon") as mock_toon:
            mock_toon.encode.return_value = "products:\n  f: id name price\n  1 Product A 10.0\n  2 Product B 20.0"

            result = _json_to_toon(data)

            mock_toon.encode.assert_called_once_with(data)
            assert "products:" in result
            assert "f: id name price" in result

    def test_json_to_toon_dict_with_key(self):
        """Test converting dict with specified key."""
        data = {"products": [{"id": 1, "name": "Product A"}, {"id": 2, "name": "Product B"}], "total": 2}

        with patch("prompture.core.toon") as mock_toon:
            mock_toon.encode.return_value = "test_output"

            result = _json_to_toon(data, data_key="products")

            mock_toon.encode.assert_called_once_with(data["products"])
            assert result == "test_output"

    def test_json_to_toon_dict_auto_detect(self):
        """Test auto-detecting array in dict."""
        data = {"metadata": {"page": 1}, "items": [{"id": 1}, {"id": 2}]}

        with patch("prompture.core.toon") as mock_toon:
            mock_toon.encode.return_value = "auto_detected"

            result = _json_to_toon(data)

            mock_toon.encode.assert_called_once_with([{"id": 1}, {"id": 2}])
            assert result == "auto_detected"

    def test_json_to_toon_empty_list_error(self):
        """Test error on empty list."""
        with pytest.raises(ValueError, match="Array data cannot be empty"):
            _json_to_toon([])

    def test_json_to_toon_non_dict_items_error(self):
        """Test error when array items are not dicts."""
        data = [1, 2, 3]  # Not dicts

        with pytest.raises(ValueError, match="All items in array must be dictionaries"):
            _json_to_toon(data)

    def test_json_to_toon_missing_key_error(self):
        """Test error when specified key doesn't exist."""
        data = {"other_key": [{"id": 1}]}

        with pytest.raises(ValueError, match="Key 'products' not found"):
            _json_to_toon(data, data_key="products")

    def test_json_to_toon_no_array_found_error(self):
        """Test error when no array found in dict."""
        data = {"count": 5, "status": "ok"}

        with pytest.raises(ValueError, match="No array found in data"):
            _json_to_toon(data)

    def test_json_to_toon_missing_dependency_error(self):
        """Test error when python-toon not installed."""
        data = [{"id": 1}]

        with patch("prompture.core.toon", None):
            with pytest.raises(RuntimeError, match="python-toon.*not installed"):
                _json_to_toon(data)

    def test_json_to_toon_encode_failure(self):
        """Test error when TOON encoding fails."""
        data = [{"id": 1}]

        with patch("prompture.core.toon") as mock_toon:
            mock_toon.encode.side_effect = Exception("TOON error")

            with pytest.raises(ValueError, match="Failed to convert data to TOON format"):
                _json_to_toon(data)


class TestDataFrameToToon:
    """Test _dataframe_to_toon helper function."""

    @patch("prompture.core.toon")
    def test_dataframe_to_toon_success(self, mock_toon):
        """Test successful DataFrame to TOON conversion."""
        # Mock pandas
        with patch.dict("sys.modules", {"pandas": Mock()}):
            import sys

            sys.modules["pandas"]

            # Create mock DataFrame
            df_mock = Mock()
            df_mock.empty = False
            df_mock.to_dict.return_value = [{"id": 1, "name": "test"}]

            mock_toon.encode.return_value = "toon_output"

            result = _dataframe_to_toon(df_mock)

            df_mock.to_dict.assert_called_once_with("records")
            mock_toon.encode.assert_called_once_with([{"id": 1, "name": "test"}])
            assert result == "toon_output"

    def test_dataframe_to_toon_pandas_missing(self):
        """Test error when pandas not installed."""
        df_mock = Mock()

        with patch.dict("sys.modules", {"pandas": None}), patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(RuntimeError, match="pandas.*not installed"):
                _dataframe_to_toon(df_mock)

    @patch("prompture.core.toon", None)
    def test_dataframe_to_toon_toon_missing(self):
        """Test error when python-toon not installed."""
        with patch.dict("sys.modules", {"pandas": Mock()}):
            df_mock = Mock()

            with pytest.raises(RuntimeError, match="python-toon.*not installed"):
                _dataframe_to_toon(df_mock)

    @patch("prompture.core.toon")
    def test_dataframe_to_toon_empty_df(self, mock_toon):
        """Test error on empty DataFrame."""
        with patch.dict("sys.modules", {"pandas": Mock()}):
            import sys

            sys.modules["pandas"]

            df_mock = Mock()
            df_mock.empty = True

            with pytest.raises(ValueError, match="DataFrame cannot be empty"):
                _dataframe_to_toon(df_mock)

    @patch("prompture.core.toon")
    def test_dataframe_to_toon_wrong_type(self, mock_toon):
        """Test error on non-DataFrame input."""
        with patch.dict("sys.modules", {"pandas": Mock()}):
            import sys

            pd_mock = sys.modules["pandas"]
            pd_mock.DataFrame = Mock  # Set DataFrame class for isinstance check

            with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
                _dataframe_to_toon("not_a_dataframe")


class TestCalculateTokenSavings:
    """Test _calculate_token_savings helper function."""

    def test_calculate_token_savings_basic(self):
        """Test basic token savings calculation."""
        json_text = "a" * 100  # 100 characters
        toon_text = "b" * 40  # 40 characters

        result = _calculate_token_savings(json_text, toon_text)

        assert result["json_characters"] == 100
        assert result["toon_characters"] == 40
        assert result["saved_characters"] == 60
        assert result["estimated_json_tokens"] == 25  # 100/4
        assert result["estimated_toon_tokens"] == 10  # 40/4
        assert result["estimated_saved_tokens"] == 15
        assert result["percentage_saved"] == 60.0

    def test_calculate_token_savings_no_savings(self):
        """Test when TOON is same size as JSON."""
        json_text = "test"
        toon_text = "test"

        result = _calculate_token_savings(json_text, toon_text)

        assert result["saved_characters"] == 0
        assert result["estimated_saved_tokens"] == 0
        assert result["percentage_saved"] == 0.0

    def test_calculate_token_savings_empty_json(self):
        """Test edge case with empty JSON."""
        json_text = ""
        toon_text = "test"

        result = _calculate_token_savings(json_text, toon_text)

        assert result["percentage_saved"] == 0.0  # Can't divide by zero


class TestExtractFromData:
    """Test extract_from_data function."""

    @patch("prompture.core.get_driver_for_model")
    @patch("prompture.core.ask_for_json")
    @patch("prompture.core.toon")
    def test_extract_from_data_success(self, mock_toon, mock_ask_json, mock_get_driver):
        """Test successful data extraction."""
        # Setup mocks
        mock_toon.encode.return_value = "toon_data"
        mock_driver = Mock()
        mock_get_driver.return_value = mock_driver

        mock_ask_json.return_value = {
            "json_object": {"result": "success"},
            "json_string": '{"result": "success"}',
            "usage": {"total_tokens": 100},
        }

        data = [{"id": 1, "name": "test"}]
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        result = extract_from_data(data=data, question="What is this?", json_schema=schema, model_name="openai/gpt-4")

        # Verify calls
        mock_toon.encode.assert_called_once_with(data)
        mock_get_driver.assert_called_once_with("openai/gpt-4")
        mock_ask_json.assert_called_once()

        # Verify result structure
        assert "json_object" in result
        assert "toon_data" in result
        assert "token_savings" in result
        assert result["toon_data"] == "toon_data"
        assert result["json_object"] == {"result": "success"}

    def test_extract_from_data_empty_question(self):
        """Test error on empty question."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            extract_from_data(data=[{"id": 1}], question="", json_schema={"type": "object"}, model_name="openai/gpt-4")

    def test_extract_from_data_empty_schema(self):
        """Test error on empty schema."""
        with pytest.raises(ValueError, match="JSON schema cannot be empty"):
            extract_from_data(data=[{"id": 1}], question="What is this?", json_schema={}, model_name="openai/gpt-4")

    @patch("prompture.core._calculate_token_savings")
    @patch("prompture.core.get_driver_for_model")
    @patch("prompture.core.ask_for_json")
    @patch("prompture.core.toon")
    def test_extract_from_data_uses_data_key_for_token_savings(
        self, mock_toon, mock_ask_json, mock_get_driver, mock_calc
    ):
        """Ensure the JSON string sent to savings calc uses the specified data_key array."""
        mock_toon.encode.return_value = "toon_data"
        mock_ask_json.return_value = {"json_object": {}, "json_string": "{}", "usage": {}}
        mock_get_driver.return_value = Mock()
        mock_calc.return_value = {"saved_characters": 10}

        products = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ]
        data = {"products": products, "meta": {"page": 1}}
        schema = {"type": "object"}

        result = extract_from_data(
            data=data,
            data_key="products",
            question="List products",
            json_schema=schema,
            model_name="openai/gpt-4",
        )

        json_arg, toon_arg = mock_calc.call_args[0]
        assert json.loads(json_arg) == products  # Should use only the array under data_key
        assert toon_arg == "toon_data"
        assert result["token_savings"]["saved_characters"] == 10

    @patch("prompture.core.get_driver_for_model")
    @patch("prompture.core.ask_for_json")
    @patch("prompture.core.toon")
    def test_extract_from_data_custom_instruction_and_options(self, mock_toon, mock_ask_json, mock_get_driver):
        """Verify custom instruction template and options propagate to ask_for_json."""
        mock_toon.encode.return_value = "toon_data"
        mock_get_driver.return_value = driver = Mock()
        mock_ask_json.return_value = {
            "json_object": {"ok": True},
            "json_string": '{"ok": true}',
            "usage": {},
        }

        options = {"temperature": 0.2, "max_tokens": 50}
        instruction_template = "Custom prompt: {question}"

        extract_from_data(
            data=[{"id": 1}],
            question="Return product ids",
            json_schema={"type": "object"},
            model_name="openai/gpt-4",
            instruction_template=instruction_template,
            options=options,
        )

        kwargs = mock_ask_json.call_args.kwargs
        assert kwargs["driver"] is driver
        assert kwargs["options"] == options
        assert "Custom prompt: Return product ids" in kwargs["content_prompt"]
        assert "Data (in TOON format)" in kwargs["content_prompt"]

    @patch("prompture.core.toon")
    def test_extract_from_data_missing_array_raises(self, mock_toon):
        """Ensure ValueError surfaces when no array is present in dict input."""
        mock_toon.encode.return_value = "unused"

        with pytest.raises(ValueError, match="No array found in data"):
            extract_from_data(
                data={"count": 1},
                question="Test",
                json_schema={"type": "object"},
                model_name="openai/gpt-4",
            )


class TestExtractFromPandas:
    """Test extract_from_pandas function."""

    @patch("prompture.core.get_driver_for_model")
    @patch("prompture.core.ask_for_json")
    @patch("prompture.core.toon")
    def test_extract_from_pandas_success(self, mock_toon, mock_ask_json, mock_get_driver):
        """Test successful pandas extraction."""
        # Mock pandas DataFrame
        with patch.dict("sys.modules", {"pandas": Mock()}):
            import sys

            sys.modules["pandas"]

            df_mock = Mock()
            df_mock.empty = False
            df_mock.shape = (2, 3)
            df_mock.columns = ["id", "name", "price"]
            df_mock.dtypes = {"id": "int64", "name": "object", "price": "float64"}
            df_mock.to_dict.return_value = [{"id": 1, "name": "test", "price": 10.0}]
            df_mock.to_json.return_value = '[{"id":1,"name":"test","price":10.0}]'

            # Setup other mocks
            mock_toon.encode.return_value = "toon_data"
            mock_driver = Mock()
            mock_get_driver.return_value = mock_driver

            mock_ask_json.return_value = {
                "json_object": {"analysis": "complete"},
                "json_string": '{"analysis": "complete"}',
                "usage": {"total_tokens": 80},
            }

            schema = {"type": "object", "properties": {"analysis": {"type": "string"}}}

            result = extract_from_pandas(
                df=df_mock, question="Analyze this data", json_schema=schema, model_name="openai/gpt-4"
            )

            # Verify DataFrame methods called
            df_mock.to_dict.assert_called_once_with("records")
            df_mock.to_json.assert_called_once_with(indent=2, orient="records")

            # Verify result structure
            assert "json_object" in result
            assert "toon_data" in result
            assert "token_savings" in result
            assert "dataframe_info" in result

            df_info = result["dataframe_info"]
            assert df_info["shape"] == (2, 3)
            assert df_info["columns"] == ["id", "name", "price"]
            assert "dtypes" in df_info

    def test_extract_from_pandas_empty_question(self):
        """Test error on empty question."""
        df_mock = Mock()

        with pytest.raises(ValueError, match="Question cannot be empty"):
            extract_from_pandas(
                df=df_mock,
                question="   ",  # Just whitespace
                json_schema={"type": "object"},
                model_name="openai/gpt-4",
            )


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("prompture.core.get_driver_for_model")
    @patch("prompture.core.toon")
    def test_full_workflow_with_nested_data(self, mock_toon, mock_get_driver):
        """Test complete workflow with nested data structure."""
        # Setup mocks
        mock_toon.encode.return_value = "id name price\n1 Product_A 10.0\n2 Product_B 20.0"

        mock_driver = Mock()
        mock_driver.generate.return_value = {
            "text": '{"total": 2, "avg_price": 15.0}',
            "meta": {"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20},
        }
        mock_get_driver.return_value = mock_driver

        # Test data
        api_response = {
            "products": [{"id": 1, "name": "Product A", "price": 10.0}, {"id": 2, "name": "Product B", "price": 20.0}],
            "page": 1,
            "total_pages": 1,
        }

        schema = {"type": "object", "properties": {"total": {"type": "integer"}, "avg_price": {"type": "number"}}}

        with patch("prompture.core.ask_for_json") as mock_ask:
            mock_ask.return_value = {
                "json_object": {"total": 2, "avg_price": 15.0},
                "json_string": '{"total": 2, "avg_price": 15.0}',
                "usage": {"total_tokens": 50},
            }

            result = extract_from_data(
                data=api_response,
                data_key="products",
                question="Count products and calculate average price",
                json_schema=schema,
                model_name="openai/gpt-4",
            )

            # Verify the products array was extracted and converted
            mock_toon.encode.assert_called_once_with(api_response["products"])

            # Verify result contains expected data
            assert result["json_object"]["total"] == 2
            assert result["json_object"]["avg_price"] == 15.0
            assert "token_savings" in result
            assert "toon_data" in result


class TestRealLLMIntegration:
    """End-to-end tests that hit a real LLM via the integration driver."""

    @pytest.mark.integration
    def test_extract_from_data_with_real_llm(self, integration_driver):
        from tests.conftest import DEFAULT_MODEL

        data = [
            {"id": 1, "name": "Laptop", "price": 999.99},
            {"id": 2, "name": "Book", "price": 19.99},
        ]

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "product_names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["count", "product_names"],
        }

        # Reuse the configured integration driver so we don't re-instantiate drivers here.
        with patch("prompture.core.get_driver_for_model", return_value=integration_driver):
            result = extract_from_data(
                data=data,
                question="Return the product count and the list of product names.",
                json_schema=schema,
                model_name=DEFAULT_MODEL,
            )

        obj = result["json_object"]
        assert isinstance(obj, dict)
        assert isinstance(obj.get("count"), int)
        assert obj.get("count", 0) >= 2
        assert isinstance(obj.get("product_names"), list)
        assert len(obj["product_names"]) >= 2

    @pytest.mark.integration
    def test_extract_from_pandas_with_real_llm(self, integration_driver):
        pd = pytest.importorskip("pandas")
        from tests.conftest import DEFAULT_MODEL

        df = pd.DataFrame(
            [
                {"product": "Laptop", "price": 999.99},
                {"product": "Book", "price": 19.99},
            ]
        )

        schema = {
            "type": "object",
            "properties": {
                "most_expensive": {"type": "string"},
                "price_range": {"type": "number"},
            },
            "required": ["most_expensive"],
        }

        with patch("prompture.core.get_driver_for_model", return_value=integration_driver):
            result = extract_from_pandas(
                df=df,
                question="Identify the most expensive product and the price range.",
                json_schema=schema,
                model_name=DEFAULT_MODEL,
            )

        obj = result["json_object"]
        assert isinstance(obj, dict)
        assert isinstance(obj.get("most_expensive"), str)
        # price_range may be absent if model decides; only check type when present
        if "price_range" in obj:
            assert isinstance(obj["price_range"], (int, float))

    @pytest.mark.integration
    def test_find_specific_record_in_json(self, integration_driver):
        from tests.conftest import DEFAULT_MODEL

        data = [
            {"id": 1, "name": "Laptop", "price": 999.99},
            {"id": 2, "name": "Book", "price": 19.99},
        ]

        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["id", "name"],
        }

        with patch("prompture.core.get_driver_for_model", return_value=integration_driver):
            result = extract_from_data(
                data=data,
                question="Return the record with id = 2.",
                json_schema=schema,
                model_name=DEFAULT_MODEL,
            )

        obj = result["json_object"]
        assert obj.get("id") == 2
        assert isinstance(obj.get("name"), str)
        if "price" in obj:
            assert isinstance(obj["price"], (int, float))

    @pytest.mark.integration
    def test_find_specific_record_in_pandas(self, integration_driver):
        pd = pytest.importorskip("pandas")
        from tests.conftest import DEFAULT_MODEL

        df = pd.DataFrame(
            [
                {"id": 1, "city": "Miami", "temp": 80},
                {"id": 2, "city": "Boston", "temp": 60},
            ]
        )

        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "city": {"type": "string"},
                "temp": {"type": "number"},
            },
            "required": ["id", "city"],
        }

        with patch("prompture.core.get_driver_for_model", return_value=integration_driver):
            result = extract_from_pandas(
                df=df,
                question="Find the row where id = 2 and return that record.",
                json_schema=schema,
                model_name=DEFAULT_MODEL,
            )

        obj = result["json_object"]
        assert obj.get("id") == 2
        assert isinstance(obj.get("city"), str)
        if "temp" in obj:
            assert isinstance(obj["temp"], (int, float))
