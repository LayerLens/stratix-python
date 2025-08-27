from datetime import timedelta
from unittest.mock import Mock, patch

import httpx
import pytest

from layerlens import Atlas
from layerlens.models import (
    Model,
    Result,
    Benchmark,
    Evaluation,
    EvaluationStatus,
    CreateEvaluationsResponse,
)


class TestAtlasIntegration:
    """Integration tests for full Atlas API workflows."""

    @pytest.fixture
    def atlas_client(self):
        """Create Atlas client with mocked dependencies."""
        return Atlas(api_key="test-api-key")

    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for testing."""
        return {
            "id": "model-gpt4",
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
    def sample_benchmark_data(self):
        """Sample benchmark data for testing."""
        return {
            "id": "benchmark-mmlu",
            "key": "mmlu",
            "name": "MMLU",
            "full_description": "Massive Multitask Language Understanding",
            "language": "english",
            "categories": ["reasoning", "knowledge"],
            "subsets": ["math", "science", "history"],
            "prompt_count": 15908,
            "deprecated": False,
        }

    @pytest.fixture
    def sample_evaluation_data(self):
        """Sample evaluation data for testing."""
        return {
            "id": "eval-12345",
            "status": "success",
            "status_description": "Evaluation completed successfully",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-gpt4",
            "model_name": "GPT-4",
            "model_key": "gpt-4",
            "model_company": "OpenAI",
            "dataset_id": "benchmark-mmlu",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "readability_score": 0.85,
            "toxicity_score": 0.02,
            "ethics_score": 0.92,
            "accuracy": 0.89,
        }

    @pytest.fixture
    def sample_result_data(self):
        """Sample result data for testing."""
        return {
            "subset": "mathematics",
            "prompt": "What is the derivative of x^2?",
            "result": "2x",
            "truth": "2x",
            "duration": timedelta(seconds=2.5),
            "score": 1.0,
            "metrics": {"accuracy": 1.0, "confidence": 0.95},
        }


class TestCompleteEvaluationWorkflow:
    """Test complete evaluation workflow from start to finish."""

    @pytest.fixture
    def atlas_client(self):
        """Atlas client for workflow testing."""
        mock_org = Mock()
        mock_org.id = "org-123"
        mock_org.projects = [Mock(id="proj-456")]

        with patch("layerlens.Atlas._get_organization", return_value=mock_org):
            return Atlas(api_key="workflow-test-key")

    def test_complete_evaluation_workflow(self, atlas_client):
        """Test complete workflow: get models/benchmarks -> create evaluation -> get results."""

        # Mock data
        model_data = {
            "id": "model-123",
            "key": "gpt-4",
            "name": "GPT-4",
            "company": "OpenAI",
            "description": "LLM",
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

        benchmark_data = {
            "id": "bench-456",
            "key": "mmlu",
            "name": "MMLU",
            "full_description": "MMLU benchmark",
            "language": "english",
            "categories": ["reasoning"],
            "subsets": ["math"],
            "prompt_count": 1000,
            "deprecated": False,
        }

        evaluation_data = {
            "id": "eval-789",
            "status": "success",
            "status_description": "Done",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-123",
            "dataset_id": "bench-456",
            "average_duration": 2500,
            "readability_score": 0.85,
            "toxicity_score": 0.02,
            "ethics_score": 0.92,
            "accuracy": 0.89,
        }

        result_data = {
            "subset": "math",
            "prompt": "2+2=?",
            "result": "4",
            "truth": "4",
            "duration": timedelta(seconds=1.5),
            "score": 1.0,
            "metrics": {"accuracy": 1.0},
        }

        # Create model objects
        model = Model(**model_data)
        benchmark = Benchmark(**benchmark_data)
        evaluation = Evaluation(**evaluation_data)
        result = Result(**result_data)

        # Mock responses
        evaluations_response = CreateEvaluationsResponse(data=[evaluation])

        with patch.object(atlas_client, "get_cast") as mock_get, patch.object(atlas_client, "post_cast") as mock_post:
            # Configure mocks for the workflow
            mock_get.return_value = {
                "evaluation_id": "eval-789",
                "results": [result_data],
                "metrics": {
                    "total_count": 1,
                    "min_toxicity_score": 0.02,
                    "max_toxicity_score": 0.02,
                    "min_readability_score": 0.85,
                    "max_readability_score": 0.85,
                },
            }  # Get results - raw API response
            mock_post.return_value = evaluations_response  # Create evaluation

            # Step 1: Create evaluation directly (Atlas client doesn't expose models/benchmarks resources)
            created_evaluation = atlas_client.evaluations.create(model=model, benchmark=benchmark)
            assert created_evaluation.id == "eval-789"
            assert created_evaluation.status == EvaluationStatus.SUCCESS

            # Step 2: Get evaluation results
            results = atlas_client.results.get(evaluation=created_evaluation)
            assert len(results.results) == 1
            assert results.results[0].score == 1.0
            assert results.results[0].subset == "math"

            # Verify all API calls were made correctly
            assert mock_get.call_count == 1  # Only results call
            assert mock_post.call_count == 1

            # Verify specific API calls
            get_calls = mock_get.call_args_list
            assert "/results" in get_calls[0][0][0]

            post_call = mock_post.call_args_list[0]
            assert "/evaluations" in post_call[0][0]

    def test_workflow_with_error_handling(self, atlas_client):
        """Test workflow handles errors gracefully."""
        from layerlens._exceptions import NotFoundError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        with patch.object(atlas_client, "get_cast") as mock_get:
            # Mock API error when getting results
            api_error = NotFoundError("Results not found", response=mock_response, body=None)
            mock_get.side_effect = api_error

            # Verify error is propagated
            with pytest.raises(NotFoundError):
                atlas_client.results.get_by_id(evaluation_id="test-eval")

    def test_workflow_with_custom_timeouts(self, atlas_client):
        """Test workflow respects custom timeout settings."""
        result_data = {
            "subset": "test",
            "prompt": "test",
            "result": "test",
            "truth": "test",
            "duration": timedelta(seconds=1.0),
            "score": 1.0,
            "metrics": {"accuracy": 1.0},
        }

        with patch.object(atlas_client, "get_cast") as mock_get:
            mock_get.return_value = {
                "evaluation_id": "test-eval",
                "results": [result_data],
                "metrics": {
                    "total_count": 1,
                    "min_toxicity_score": 0.0,
                    "max_toxicity_score": 0.1,
                    "min_readability_score": 0.8,
                    "max_readability_score": 0.9,
                },
            }

            # Test with custom timeout
            custom_timeout = httpx.Timeout(30.0)
            results = atlas_client.results.get_by_id(evaluation_id="test-eval", timeout=custom_timeout)

            assert len(results.results) == 1

            # Verify timeout was passed correctly
            call_args = mock_get.call_args
            assert call_args.kwargs["timeout"] is custom_timeout


class TestResourceInteraction:
    """Test interactions between different resources."""

    @pytest.fixture
    def atlas_client(self):
        """Atlas client for resource interaction testing."""
        mock_org = Mock()
        mock_org.id = "org-123"
        mock_org.projects = [Mock(id="proj-456")]

        with patch("layerlens.Atlas._get_organization", return_value=mock_org):
            return Atlas(api_key="interaction-test-key")

    def test_evaluation_creation_with_model_and_benchmark_objects(self, atlas_client):
        """Test creating evaluation using model and benchmark objects."""

        # Create model and benchmark objects
        model_data = {
            "id": "model-abc",
            "key": "claude-3",
            "name": "Claude 3",
            "company": "Anthropic",
            "description": "Claude 3",
            "released_at": 1709251200,
            "parameters": 5e11,
            "modality": "text",
            "context_length": 100000,
            "architecture_type": "transformer",
            "license": "proprietary",
            "open_weights": False,
            "region": "us-west-2",
            "deprecated": False,
        }

        benchmark_data = {
            "id": "bench-xyz",
            "key": "hellaswag",
            "name": "HellaSwag",
            "full_description": "HellaSwag benchmark",
            "language": "english",
            "categories": ["reasoning"],
            "subsets": ["commonsense"],
            "prompt_count": 10042,
            "deprecated": False,
        }

        evaluation_data = {
            "id": "eval-interaction",
            "status": "in-progress",
            "status_description": "Submitted",
            "submitted_at": 1640995200,
            "finished_at": 0,
            "model_id": "model-abc",
            "dataset_id": "bench-xyz",
            "average_duration": 0,
            "readability_score": 0.0,
            "toxicity_score": 0.0,
            "ethics_score": 0.0,
            "accuracy": 0.0,
        }

        model = Model(**model_data)
        benchmark = Benchmark(**benchmark_data)
        evaluation = Evaluation(**evaluation_data)

        evaluations_response = CreateEvaluationsResponse(data=[evaluation])

        with patch.object(atlas_client, "post_cast") as mock_post:
            mock_post.return_value = evaluations_response

            # Create evaluation using model and benchmark keys
            created_evaluation = atlas_client.evaluations.create(model=model, benchmark=benchmark)

            assert created_evaluation.id == "eval-interaction"
            assert created_evaluation.model_id == model.id
            assert created_evaluation.benchmark_id == benchmark.id

            # Verify API call
            call_args = mock_post.call_args
            body = call_args.kwargs["body"][0]
            assert body["model_id"] == model.id
            assert body["dataset_id"] == benchmark.id

    def test_results_analysis_workflow(self, atlas_client):
        """Test analyzing results from multiple evaluations."""

        # Create multiple result objects
        results_data = [
            {
                "subset": "math",
                "prompt": "2+2=?",
                "result": "4",
                "truth": "4",
                "duration": timedelta(seconds=1.0),
                "score": 1.0,
                "metrics": {"accuracy": 1.0},
            },
            {
                "subset": "math",
                "prompt": "3*3=?",
                "result": "9",
                "truth": "9",
                "duration": timedelta(seconds=1.2),
                "score": 1.0,
                "metrics": {"accuracy": 1.0},
            },
            {
                "subset": "reading",
                "prompt": "What is the main idea?",
                "result": "Education",
                "truth": "Learning",
                "duration": timedelta(seconds=2.8),
                "score": 0.7,
                "metrics": {"accuracy": 0.7},
            },
        ]

        results = [Result(**data) for data in results_data]

        with patch.object(atlas_client, "get_cast") as mock_get:
            mock_get.return_value = {
                "evaluation_id": "test-eval",
                "results": results_data,
                "metrics": {
                    "total_count": 3,
                    "min_toxicity_score": 0.0,
                    "max_toxicity_score": 0.1,
                    "min_readability_score": 0.7,
                    "max_readability_score": 0.9,
                },
            }

            # Get results
            evaluation_results = atlas_client.results.get_by_id(evaluation_id="test-eval")

            # Analyze results
            math_results = [r for r in evaluation_results.results if r.subset == "math"]
            reading_results = [r for r in evaluation_results.results if r.subset == "reading"]

            assert len(math_results) == 2
            assert len(reading_results) == 1

            # Calculate average scores
            math_avg = sum(r.score for r in math_results) / len(math_results)
            reading_avg = sum(r.score for r in reading_results) / len(reading_results)

            assert math_avg == 1.0
            assert reading_avg == 0.7

            # Calculate average duration
            avg_duration = sum((r.duration.total_seconds() for r in evaluation_results.results), 0.0) / len(
                evaluation_results.results
            )
            expected_avg = (1.0 + 1.2 + 2.8) / 3
            assert abs(avg_duration - expected_avg) < 0.01


class TestAtlasClientProperties:
    """Test Atlas client resource properties and access."""

    @pytest.fixture
    def mock_org(self):
        org = Mock()
        org.id = "org-123"
        org.projects = [Mock(id="proj-456")]
        return org

    def test_client_has_all_resource_properties(self, mock_org):
        """Atlas client exposes all resource properties."""
        with patch("layerlens.Atlas._get_organization", return_value=mock_org):
            client = Atlas(api_key="property-test-key")

        # Verify available resource properties exist
        assert hasattr(client, "evaluations")
        assert hasattr(client, "results")

        # Verify they are the correct types
        from layerlens.resources.results import Results
        from layerlens.resources.evaluations import Evaluations

        assert isinstance(client.evaluations, Evaluations)
        assert isinstance(client.results, Results)

    def test_resource_properties_share_same_client(self, mock_org):
        """All resource properties share the same client instance."""
        with patch("layerlens.Atlas._get_organization", return_value=mock_org):
            client = Atlas(api_key="shared-client-test")

        # Verify all resources use the same client
        assert client.evaluations._client is client
        assert client.results._client is client


class TestConcurrentOperations:
    """Test concurrent operations and resource independence."""

    @pytest.fixture
    def mock_org1(self):
        org = Mock()
        org.id = "org-123"
        org.projects = [Mock(id="proj-456")]
        return org

    @pytest.fixture
    def mock_org2(self):
        org = Mock()
        org.id = "org-456"
        org.projects = [Mock(id="proj-123")]
        return org

    def test_multiple_atlas_clients_independent(self, mock_org1, mock_org2):
        """Multiple Atlas client instances operate independently."""

        with patch("layerlens.Atlas._get_organization", return_value=mock_org1):
            client1 = Atlas(api_key="client-1-key")

        with patch("layerlens.Atlas._get_organization", return_value=mock_org2):
            client2 = Atlas(api_key="client-2-key")

        # Verify clients are independent
        assert client1.api_key != client2.api_key

        # Verify resources are independent
        assert client1.evaluations._client is not client2.evaluations._client
        assert client1.results._client is not client2.results._client

    def test_resource_operations_isolated(self, mock_org1, mock_org2):
        """Operations on different client resources are isolated."""

        with patch("layerlens.Atlas._get_organization", return_value=mock_org1):
            client1 = Atlas(api_key="iso-test-1")

        with patch("layerlens.Atlas._get_organization", return_value=mock_org2):
            client2 = Atlas(api_key="iso-test-2")

        result_data = {
            "subset": "test",
            "prompt": "test",
            "result": "test",
            "truth": "test",
            "duration": timedelta(seconds=1.0),
            "score": 1.0,
            "metrics": {"accuracy": 1.0},
        }

        with patch.object(client1, "get_cast") as mock_get1, patch.object(client2, "get_cast") as mock_get2:
            mock_get1.return_value = {
                "evaluation_id": "test-eval",
                "results": [result_data],
                "metrics": {
                    "total_count": 1,
                    "min_toxicity_score": 0.0,
                    "max_toxicity_score": 0.1,
                    "min_readability_score": 0.8,
                    "max_readability_score": 0.9,
                },
            }
            mock_get2.return_value = {
                "evaluation_id": "test-eval",
                "results": [result_data],
                "metrics": {
                    "total_count": 1,
                    "min_toxicity_score": 0.0,
                    "max_toxicity_score": 0.1,
                    "min_readability_score": 0.8,
                    "max_readability_score": 0.9,
                },
            }

            # Make calls on both clients
            results1 = client1.results.get_by_id(evaluation_id="eval-1")
            results2 = client2.results.get_by_id(evaluation_id="eval-2")

            # Verify both calls succeeded
            assert results1 is not None
            assert len(results1.results) == 1
            assert results2 is not None
            assert len(results2.results) == 1

            # Verify calls were made to correct clients
            mock_get1.assert_called_once()
            mock_get2.assert_called_once()

            # Verify different parameters were used
            call1_params = mock_get1.call_args.kwargs["params"]
            call2_params = mock_get2.call_args.kwargs["params"]
            assert call1_params["evaluation_id"] == "eval-1"
            assert call2_params["evaluation_id"] == "eval-2"


class TestErrorPropagation:
    """Test error propagation through full workflows."""

    @pytest.fixture
    def mock_org(self):
        org = Mock()
        org.id = "org-123"
        org.projects = [Mock(id="proj-456")]
        return org

    def test_evaluation_workflow_error_propagation(self, mock_org):
        """Errors in evaluation workflow are properly propagated."""
        from layerlens._exceptions import APIStatusError, APIConnectionError

        # Create model and benchmark objects
        model_data = {
            "id": "model-abc",
            "key": "claude-3",
            "name": "Claude 3",
            "company": "Anthropic",
            "description": "Claude 3",
            "released_at": 1709251200,
            "parameters": 5e11,
            "modality": "text",
            "context_length": 100000,
            "architecture_type": "transformer",
            "license": "proprietary",
            "open_weights": False,
            "region": "us-west-2",
            "deprecated": False,
        }

        benchmark_data = {
            "id": "bench-xyz",
            "key": "hellaswag",
            "name": "HellaSwag",
            "full_description": "HellaSwag benchmark",
            "language": "english",
            "categories": ["reasoning"],
            "subsets": ["commonsense"],
            "prompt_count": 10042,
            "deprecated": False,
        }

        model = Model(**model_data)
        benchmark = Benchmark(**benchmark_data)

        with patch("layerlens.Atlas._get_organization", return_value=mock_org):
            client = Atlas(api_key="error-test-key")

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}

        # Test different types of errors
        api_error = APIStatusError("Server Error", response=mock_response, body=None)
        connection_error = APIConnectionError(request=Mock())

        with patch.object(client, "get_cast") as mock_get, patch.object(client, "post_cast") as mock_post:
            # Test API error in results.get
            mock_get.side_effect = api_error
            with pytest.raises(APIStatusError):
                client.results.get_by_id(evaluation_id="test-eval")

            # Test connection error in evaluations.create
            mock_post.side_effect = connection_error
            with pytest.raises(APIConnectionError):
                client.evaluations.create(model=model, benchmark=benchmark)

            # Verify errors didn't interfere with each other
            assert mock_get.called
            assert mock_post.called
