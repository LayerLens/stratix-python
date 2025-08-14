from unittest.mock import Mock

import httpx
import pytest

from atlas._models import Model, Models as ModelsData, CustomModel
from atlas._constants import DEFAULT_TIMEOUT
from atlas.resources.models.models import Models


class TestModels:
    """Test Models resource API methods."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def models_resource(self, mock_client):
        """Models resource instance."""
        return Models(mock_client)

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for testing."""
        return {
            "id": "model-123",
            "key": "gpt-4",
            "name": "GPT-4",
            "company": "OpenAI",
            "description": "Large language model",
            "released_at": 1679875200,
            "parameters": 1.76e12,
            "modality": "text",
            "context_length": 8192,
            "architecture_type": "transformer",
            "license": "proprietary",
            "open_weights": False,
            "region": "us-east-1",
            "deprecated": False,
        }

    @pytest.fixture
    def sample_custom_model_data(self):
        """Sample custom model data for testing."""
        return {
            "id": "custom-model-456",
            "key": "my-model",
            "name": "My Custom Model",
            "description": "Custom model description",
            "max_tokens": 4096,
            "api_url": "https://api.example.com/v1/chat",
            "disabled": False,
        }

    @pytest.fixture
    def mock_public_models_response(self, sample_model_data):
        """Mock ModelsData response with public models."""
        model = Model(**sample_model_data)
        return ModelsData(models=[model])

    @pytest.fixture
    def mock_custom_models_response(self, sample_custom_model_data):
        """Mock ModelsData response with custom models."""
        custom_model = CustomModel(**sample_custom_model_data)
        return ModelsData(models=[custom_model])

    def test_models_initialization(self, mock_client):
        """Models resource initializes correctly."""
        models = Models(mock_client)

        assert models._client is mock_client
        assert models._get is mock_client.get_cast

    def test_get_public_models_success(self, models_resource, mock_public_models_response):
        """get method returns public models successfully."""
        models_resource._get.return_value = mock_public_models_response

        result = models_resource.get(type="public")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Model)
        assert result[0].name == "GPT-4"
        assert result[0].key == "gpt-4"
        assert result[0].company == "OpenAI"

    def test_get_custom_models_success(self, models_resource, mock_custom_models_response):
        """get method returns custom models successfully."""
        models_resource._get.return_value = mock_custom_models_response

        result = models_resource.get(type="custom")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], CustomModel)
        assert result[0].name == "My Custom Model"
        assert result[0].key == "my-model"
        assert result[0].api_url == "https://api.example.com/v1/chat"

    def test_get_models_request_parameters_public(self, models_resource, mock_public_models_response):
        """get method makes correct API request for public models."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(type="public")

        models_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/models",
            params={"type": "public"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=ModelsData,
        )

    def test_get_models_request_parameters_custom(self, models_resource, mock_custom_models_response):
        """get method makes correct API request for custom models."""
        models_resource._get.return_value = mock_custom_models_response

        models_resource.get(type="custom")

        models_resource._get.assert_called_once_with(
            "/organizations/org-123/projects/proj-456/models",
            params={"type": "custom"},
            timeout=DEFAULT_TIMEOUT,
            cast_to=ModelsData,
        )

    def test_get_models_with_custom_timeout(self, models_resource, mock_public_models_response):
        """get method accepts custom timeout."""
        models_resource._get.return_value = mock_public_models_response
        custom_timeout = 60.0

        models_resource.get(type="public", timeout=custom_timeout)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_get_models_with_httpx_timeout(self, models_resource, mock_public_models_response):
        """get method accepts httpx.Timeout object."""
        models_resource._get.return_value = mock_public_models_response
        custom_timeout = httpx.Timeout(60.0)

        models_resource.get(type="public", timeout=custom_timeout)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    def test_get_models_none_response(self, models_resource):
        """get method returns None when response is None."""
        models_resource._get.return_value = None

        result = models_resource.get(type="public")

        assert result is None

    def test_get_models_invalid_response_type(self, models_resource):
        """get method handles non-ModelsData response gracefully."""
        models_resource._get.return_value = "invalid-response"

        result = models_resource.get(type="public")

        assert result is None

    def test_get_models_empty_response(self, models_resource):
        """get method returns empty list when no models in response."""
        empty_response = ModelsData(models=[])
        models_resource._get.return_value = empty_response

        result = models_resource.get(type="public")

        assert result == []
        assert isinstance(result, list)

    def test_get_models_multiple_items(self, models_resource, sample_model_data):
        """get method returns multiple models correctly."""
        model1 = Model(**sample_model_data)

        # Create second model with different data
        model2_data = sample_model_data.copy()
        model2_data["id"] = "model-456"
        model2_data["key"] = "gpt-3.5-turbo"
        model2_data["name"] = "GPT-3.5 Turbo"
        model2_data["parameters"] = 1.75e11
        model2 = Model(**model2_data)

        response = ModelsData(models=[model1, model2])
        models_resource._get.return_value = response

        result = models_resource.get(type="public")

        assert len(result) == 2
        assert result[0].key == "gpt-4"
        assert result[1].key == "gpt-3.5-turbo"
        assert result[0].parameters == 1.76e12
        assert result[1].parameters == 1.75e11

    def test_get_models_url_construction(self, models_resource, mock_public_models_response):
        """get method constructs URL correctly with org and project IDs."""
        models_resource._client.organization_id = "custom-org"
        models_resource._client.project_id = "custom-project"
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(type="public")

        expected_url = "/organizations/custom-org/projects/custom-project/models"
        call_args = models_resource._get.call_args
        assert call_args[0][0] == expected_url

    @pytest.mark.parametrize("model_type", ["public", "custom"])
    def test_get_models_type_parameter(self, models_resource, model_type):
        """get method accepts both public and custom types."""
        models_resource._get.return_value = ModelsData(models=[])

        models_resource.get(type=model_type)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["params"]["type"] == model_type

    def test_get_models_cast_to_parameter(self, models_resource, mock_public_models_response):
        """get method specifies correct cast_to parameter."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(type="public")

        call_args = models_resource._get.call_args
        assert call_args.kwargs["cast_to"] is ModelsData

    def test_get_models_timeout_default(self, models_resource, mock_public_models_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(type="public")

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_models_with_none_timeout(self, models_resource, mock_public_models_response):
        """get method accepts None timeout."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(type="public", timeout=None)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_models_model_attributes(self, models_resource, mock_public_models_response):
        """get method preserves all model attributes correctly."""
        models_resource._get.return_value = mock_public_models_response

        result = models_resource.get(type="public")
        model = result[0]

        assert model.context_length == 8192
        assert model.open_weights is False
        assert model.deprecated is False
        assert model.region == "us-east-1"
        assert model.license == "proprietary"
        assert model.architecture_type == "transformer"
        assert model.modality == "text"

    def test_get_models_custom_model_attributes(self, models_resource, mock_custom_models_response):
        """get method preserves all custom model attributes correctly."""
        models_resource._get.return_value = mock_custom_models_response

        result = models_resource.get(type="custom")
        custom_model = result[0]

        assert custom_model.max_tokens == 4096
        assert custom_model.disabled is False
        assert custom_model.api_url == "https://api.example.com/v1/chat"


class TestModelsErrorHandling:
    """Test error handling in Models resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def models_resource(self, mock_client):
        """Models resource instance."""
        return Models(mock_client)

    def test_get_models_handles_api_error(self, models_resource):
        """get method propagates API errors."""
        from atlas._exceptions import APIStatusError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        api_error = APIStatusError("Internal Server Error", response=mock_response, body=None)
        models_resource._get.side_effect = api_error

        with pytest.raises(APIStatusError):
            models_resource.get(type="public")

    def test_get_models_handles_forbidden_error(self, models_resource):
        """get method propagates permission errors."""
        from atlas._exceptions import PermissionDeniedError

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        permission_error = PermissionDeniedError("Forbidden", response=mock_response, body=None)
        models_resource._get.side_effect = permission_error

        with pytest.raises(PermissionDeniedError):
            models_resource.get(type="custom")

    def test_get_models_handles_connection_error(self, models_resource):
        """get method propagates connection errors."""
        from atlas._exceptions import APIConnectionError

        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        models_resource._get.side_effect = connection_error

        with pytest.raises(APIConnectionError):
            models_resource.get(type="public")

    def test_get_models_handles_timeout_error(self, models_resource):
        """get method propagates timeout errors."""
        from atlas._exceptions import APITimeoutError

        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        models_resource._get.side_effect = timeout_error

        with pytest.raises(APITimeoutError):
            models_resource.get(type="public", timeout=5.0)


class TestModelsTyping:
    """Test type handling in Models resource."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client."""
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.get_cast = Mock()
        return client

    @pytest.fixture
    def models_resource(self, mock_client):
        """Models resource instance."""
        return Models(mock_client)

    def test_get_models_return_type_consistency(self, models_resource):
        """get method returns consistent types."""
        # Test that the method returns either a list or None
        models_resource._get.return_value = None
        result = models_resource.get(type="public")
        assert result is None

        # Test that it returns a list when successful
        models_resource._get.return_value = ModelsData(models=[])
        result = models_resource.get(type="public")
        assert isinstance(result, list)

    def test_get_models_mixed_model_types(self, models_resource):
        """get method can handle mixed model types in response."""
        # Create mixed response with both Model and CustomModel
        public_data = {
            "id": "public-123",
            "key": "gpt-4",
            "name": "GPT-4",
            "company": "OpenAI",
            "description": "Public model",
            "released_at": 1679875200,
            "parameters": 1.76e12,
            "modality": "text",
            "context_length": 8192,
            "architecture_type": "transformer",
            "license": "proprietary",
            "open_weights": False,
            "region": "us-east-1",
            "deprecated": False,
        }

        custom_data = {
            "id": "custom-456",
            "key": "my-model",
            "name": "My Custom Model",
            "description": "Custom model",
            "max_tokens": 4096,
            "api_url": "https://api.example.com/v1/chat",
            "disabled": False,
        }

        public_model = Model(**public_data)
        custom_model = CustomModel(**custom_data)

        response = ModelsData(models=[public_model, custom_model])
        models_resource._get.return_value = response

        result = models_resource.get(type="public")  # Type doesn't matter for this test

        assert len(result) == 2
        assert isinstance(result[0], Model)
        assert isinstance(result[1], CustomModel)
        assert result[0].key == "gpt-4"
        assert result[1].key == "my-model"
        assert hasattr(result[0], "parameters")  # Model-specific attribute
        assert hasattr(result[1], "max_tokens")  # CustomModel-specific attribute

    def test_get_models_large_parameters_handling(self, models_resource):
        """get method handles large parameter numbers correctly."""
        large_model_data = {
            "id": "large-model",
            "key": "claude-3-opus",
            "name": "Claude 3 Opus",
            "company": "Anthropic",
            "description": "Very large language model",
            "released_at": 1709251200,
            "parameters": 1.3e14,  # 130 trillion parameters
            "modality": "text",
            "context_length": 200000,
            "architecture_type": "transformer",
            "license": "proprietary",
            "open_weights": False,
            "region": "us-west-2",
            "deprecated": False,
        }

        large_model = Model(**large_model_data)
        response = ModelsData(models=[large_model])
        models_resource._get.return_value = response

        result = models_resource.get(type="public")

        assert len(result) == 1
        assert result[0].parameters == 1.3e14
        assert result[0].context_length == 200000
        assert isinstance(result[0].parameters, float)
        assert isinstance(result[0].context_length, int)
