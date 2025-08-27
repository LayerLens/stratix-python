from unittest.mock import Mock, call

import httpx
import pytest

from layerlens.models import CustomModel, PublicModel, ModelsResponse
from layerlens._constants import DEFAULT_TIMEOUT
from layerlens.resources.models.models import Models


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
        """Mock ModelsResponse response with public models."""
        model = PublicModel(**sample_model_data)
        return ModelsResponse(data=ModelsResponse.Data(models=[model]))

    @pytest.fixture
    def mock_custom_models_response(self, sample_custom_model_data):
        """Mock ModelsResponse response with custom models."""
        custom_model = CustomModel(**sample_custom_model_data)
        return ModelsResponse(data=ModelsResponse.Data(models=[custom_model]))

    def test_models_initialization(self, mock_client):
        """Models resource initializes correctly."""
        models = Models(mock_client)

        assert models._client is mock_client
        assert models._get is mock_client.get_cast

    def test_get_public_models_success(self, models_resource, mock_public_models_response):
        """get method returns public models successfully."""
        models_resource._get.side_effect = lambda *_, **kwargs: (
            mock_public_models_response
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        result = models_resource.get()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], PublicModel)
        assert result[0].name == "GPT-4"
        assert result[0].key == "gpt-4"
        assert result[0].company == "OpenAI"

    def test_get_custom_models_success(self, models_resource, mock_custom_models_response):
        """get method returns custom models successfully."""
        models_resource._get.side_effect = lambda *_, **kwargs: (
            mock_custom_models_response
            if kwargs.get("params", {}).get("type") == "custom"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        result = models_resource.get()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], CustomModel)
        assert result[0].name == "My Custom Model"
        assert result[0].key == "my-model"
        assert result[0].api_url == "https://api.example.com/v1/chat"

    def test_get_models_request_parameters_public(self, models_resource, mock_public_models_response):
        """get method makes correct API request for public models."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get()

        expected_calls = [
            call(
                "/organizations/org-123/projects/proj-456/models",
                params={"type": "custom"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=ModelsResponse,
            ),
            call(
                "/organizations/org-123/projects/proj-456/models",
                params={"type": "public"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=ModelsResponse,
            ),
        ]

        models_resource._get.assert_has_calls(expected_calls)

    def test_get_models_request_parameters_custom(self, models_resource, mock_custom_models_response):
        """get method makes correct API request for custom models."""
        models_resource._get.return_value = mock_custom_models_response

        models_resource.get()

        expected_calls = [
            call(
                "/organizations/org-123/projects/proj-456/models",
                params={"type": "custom"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=ModelsResponse,
            ),
            call(
                "/organizations/org-123/projects/proj-456/models",
                params={"type": "public"},
                timeout=DEFAULT_TIMEOUT,
                cast_to=ModelsResponse,
            ),
        ]

        models_resource._get.assert_has_calls(expected_calls)

    def test_get_models_with_custom_timeout(self, models_resource, mock_public_models_response):
        """get method accepts custom timeout."""
        models_resource._get.return_value = mock_public_models_response
        custom_timeout = 60.0

        models_resource.get(timeout=custom_timeout)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] == custom_timeout

    def test_get_models_with_httpx_timeout(self, models_resource, mock_public_models_response):
        """get method accepts httpx.Timeout object."""
        models_resource._get.return_value = mock_public_models_response
        custom_timeout = httpx.Timeout(60.0)

        models_resource.get(timeout=custom_timeout)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is custom_timeout

    @pytest.mark.parametrize(
        "mock_response, expected",
        [
            (None, []),  # None response
            ("invalid-response", []),  # Invalid type
            (
                ModelsResponse(data=ModelsResponse.Data(models=[])),
                [],
            ),  # Empty ModelsResponse
        ],
    )
    def test_get_models_responses(self, models_resource, mock_response, expected):
        """get method handles different types of responses correctly."""
        models_resource._get.return_value = mock_response

        result = models_resource.get()

        assert result == expected
        if isinstance(mock_response, ModelsResponse):
            assert isinstance(result, list)

    def test_get_models_multiple_items(self, models_resource, sample_model_data):
        """get method returns multiple models correctly."""
        model1 = PublicModel(**sample_model_data)

        # Create second model with different data
        model2_data = sample_model_data.copy()
        model2_data["id"] = "model-456"
        model2_data["key"] = "gpt-3.5-turbo"
        model2_data["name"] = "GPT-3.5 Turbo"
        model2_data["parameters"] = 1.75e11
        model2 = PublicModel(**model2_data)

        response = ModelsResponse(data=ModelsResponse.Data(models=[model1, model2]))

        models_resource._get.side_effect = lambda *_, **kwargs: (
            response
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        result = models_resource.get()

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

        models_resource.get()

        expected_url = "/organizations/custom-org/projects/custom-project/models"
        call_args = models_resource._get.call_args
        assert call_args[0][0] == expected_url

    def test_get_models_cast_to_parameter(self, models_resource, mock_public_models_response):
        """get method specifies correct cast_to parameter."""
        models_resource._get.side_effect = lambda *_, **kwargs: (
            mock_public_models_response
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        models_resource.get()

        call_args = models_resource._get.call_args
        assert call_args.kwargs["cast_to"] is ModelsResponse

    def test_get_models_timeout_default(self, models_resource, mock_public_models_response):
        """get method uses DEFAULT_TIMEOUT when no timeout specified."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get()

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is DEFAULT_TIMEOUT

    def test_get_models_with_none_timeout(self, models_resource, mock_public_models_response):
        """get method accepts None timeout."""
        models_resource._get.return_value = mock_public_models_response

        models_resource.get(timeout=None)

        call_args = models_resource._get.call_args
        assert call_args.kwargs["timeout"] is None

    def test_get_models_model_attributes(self, models_resource, mock_public_models_response):
        """get method preserves all model attributes correctly."""
        models_resource._get.side_effect = lambda *_, **kwargs: (
            mock_public_models_response
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        result = models_resource.get()
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

        result = models_resource.get()
        custom_model = result[0]

        assert custom_model.max_tokens == 4096
        assert custom_model.disabled is False
        assert custom_model.api_url == "https://api.example.com/v1/chat"

    def test_get_by_id_custom_benchmark(self, models_resource, sample_custom_model_data):
        """get_by_id returns CustomModel when organization_id is present."""
        sample_custom_model_data["organization_id"] = "org-123"  # Required for type detection
        models_resource._get.return_value = {"data": sample_custom_model_data}

        result = models_resource.get_by_id("custom-model-456")

        assert isinstance(result, CustomModel)
        assert result.id == "custom-model-456"
        assert result.name == "My Custom Model"

    def test_get_by_id_public_benchmark(self, models_resource, sample_model_data):
        """get_by_id returns PublicModel when organization_id is missing."""
        models_resource._get.return_value = {"data": sample_model_data}

        result = models_resource.get_by_id("model-123")

        assert isinstance(result, PublicModel)
        assert result.id == "model-123"
        assert result.name == "GPT-4"

    def test_get_by_id_invalid_response(self, models_resource):
        """get_by_id returns None for invalid responses."""
        models_resource._get.return_value = "not-a-dict"

        result = models_resource.get_by_id("invalid")

        assert result is None

    def test_get_by_key_custom_model(self, models_resource, sample_custom_model_data):
        """get_by_key returns CustomModel when key matches and organization_id is present."""
        sample_custom_model_data["organization_id"] = "org-123"
        custom_model = CustomModel(**sample_custom_model_data)
        models_resource.get = Mock(return_value=[custom_model])

        result = models_resource.get_by_key(key="my-model")

        assert isinstance(result, CustomModel)
        assert result.key == "my-model"
        assert result.name == "My Custom Model"

    def test_get_by_key_public_model(self, models_resource, sample_model_data):
        """get_by_key returns PublicModel when key matches and organization_id is missing."""
        public_model = PublicModel(**sample_model_data)
        models_resource.get = Mock(return_value=[public_model])

        result = models_resource.get_by_key(key="gpt-4")

        assert isinstance(result, PublicModel)
        assert result.key == "gpt-4"
        assert result.name == "GPT-4"

    def test_get_by_key_no_match(self, models_resource, sample_model_data):
        """get_by_key returns None if no model has the exact key."""
        public_model = PublicModel(**sample_model_data)
        models_resource.get = Mock(return_value=[public_model])

        result = models_resource.get_by_key(key="nonexistent-key")

        assert result is None

    def test_get_by_key_invalid_response(self, models_resource):
        """get_by_key returns None when get() returns None or invalid type."""
        models_resource.get = Mock(return_value=None)

        result = models_resource.get_by_key(key="some-key")

        assert result is None


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
        from layerlens._exceptions import APIStatusError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        api_error = APIStatusError("Internal Server Error", response=mock_response, body=None)
        models_resource._get.side_effect = api_error

        with pytest.raises(APIStatusError):
            models_resource.get()

    def test_get_models_handles_forbidden_error(self, models_resource):
        """get method propagates permission errors."""
        from layerlens._exceptions import PermissionDeniedError

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        permission_error = PermissionDeniedError("Forbidden", response=mock_response, body=None)
        models_resource._get.side_effect = permission_error

        with pytest.raises(PermissionDeniedError):
            models_resource.get()

    def test_get_models_handles_connection_error(self, models_resource):
        """get method propagates connection errors."""
        from layerlens._exceptions import APIConnectionError

        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        models_resource._get.side_effect = connection_error

        with pytest.raises(APIConnectionError):
            models_resource.get()

    def test_get_models_handles_timeout_error(self, models_resource):
        """get method propagates timeout errors."""
        from layerlens._exceptions import APITimeoutError

        mock_request = Mock()
        timeout_error = APITimeoutError(mock_request)
        models_resource._get.side_effect = timeout_error

        with pytest.raises(APITimeoutError):
            models_resource.get(timeout=5.0)


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

    @pytest.mark.parametrize(
        "mock_response, expected_type",
        [
            (None, list),  # None response
            (
                ModelsResponse(data=ModelsResponse.Data(models=[])),
                list,
            ),  # Empty ModelsResponse
        ],
    )
    def test_get_models_return_type_consistency(self, models_resource, mock_response, expected_type):
        """get method returns consistent types."""
        models_resource._get.return_value = mock_response

        result = models_resource.get()

        assert isinstance(result, expected_type)

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

        public_model = PublicModel(**public_data)
        custom_model = CustomModel(**custom_data)

        models_resource._get.side_effect = lambda *_, **kwargs: (
            ModelsResponse(data=ModelsResponse.Data(models=[public_model]))
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[custom_model]))
        )

        result = models_resource.get()  # Type doesn't matter for this test

        assert len(result) == 2
        assert isinstance(result[0], CustomModel)
        assert isinstance(result[1], PublicModel)
        assert result[0].key == "my-model"
        assert result[1].key == "gpt-4"
        assert hasattr(result[0], "max_tokens")  # CustomModel-specific attribute
        assert hasattr(result[1], "parameters")  # PublicModel-specific attribute

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

        large_model = PublicModel(**large_model_data)
        response = ModelsResponse(data=ModelsResponse.Data(models=[large_model]))
        models_resource._get.side_effect = lambda *_, **kwargs: (
            response
            if kwargs.get("params", {}).get("type") == "public"
            else ModelsResponse(data=ModelsResponse.Data(models=[]))
        )

        result = models_resource.get()

        assert len(result) == 1
        assert result[0].parameters == 1.3e14
        assert result[0].context_length == 200000
        assert isinstance(result[0].parameters, float)
        assert isinstance(result[0].context_length, int)
