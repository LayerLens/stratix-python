from datetime import timedelta

import pytest
from pydantic import ValidationError

from atlas._models import (
    Model,
    Models,
    Result,
    Results,
    Benchmark,
    Benchmarks,
    Evaluation,
    Pagination,
    CustomModel,
    Evaluations,
    ResultMetrics,
    CustomBenchmark,
)


class TestEvaluation:
    """Test Evaluation model validation and serialization."""

    @pytest.fixture
    def valid_evaluation_data(self):
        """Valid evaluation data for testing."""
        return {
            "id": "eval-123",
            "status": "completed",
            "status_description": "Evaluation completed successfully",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "model_name": "GPT-4",
            "model_key": "gpt-4",
            "model_company": "OpenAI",
            "dataset_id": "dataset-789",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "readability_score": 0.85,
            "toxicity_score": 0.02,
            "ethics_score": 0.92,
            "accuracy": 0.89,
        }

    def test_evaluation_creation_with_valid_data(self, valid_evaluation_data):
        """Evaluation model creates successfully with valid data."""
        evaluation = Evaluation(**valid_evaluation_data)

        assert evaluation.id == "eval-123"
        assert evaluation.status == "completed"
        assert evaluation.model_name == "GPT-4"
        assert evaluation.accuracy == 0.89
        assert evaluation.readability_score == 0.85

    def test_evaluation_field_types(self, valid_evaluation_data):
        """Evaluation model enforces correct field types."""
        evaluation = Evaluation(**valid_evaluation_data)

        assert isinstance(evaluation.id, str)
        assert isinstance(evaluation.submitted_at, int)
        assert isinstance(evaluation.readability_score, float)
        assert isinstance(evaluation.accuracy, float)

    def test_evaluation_validation_errors(self, valid_evaluation_data):
        """Evaluation model validates field types and requirements."""
        # Test string field with wrong type
        invalid_data = valid_evaluation_data.copy()
        invalid_data["id"] = 123
        with pytest.raises(ValidationError):
            Evaluation(**invalid_data)

        # Test int field with wrong type
        invalid_data = valid_evaluation_data.copy()
        invalid_data["submitted_at"] = "not-an-int"
        with pytest.raises(ValidationError):
            Evaluation(**invalid_data)

    def test_evaluation_missing_required_fields(self):
        """Evaluation model requires all fields."""
        incomplete_data = {"id": "eval-123", "status": "pending"}

        with pytest.raises(ValidationError) as exc_info:
            Evaluation(**incomplete_data)  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert len(errors) > 5

    def test_evaluation_json_serialization(self, valid_evaluation_data):
        """Evaluation model serializes to JSON correctly."""
        evaluation = Evaluation(**valid_evaluation_data)
        json_data = evaluation.model_dump()

        assert json_data["id"] == "eval-123"
        assert json_data["accuracy"] == 0.89
        assert isinstance(json_data, dict)


class TestEvaluations:
    """Test Evaluations collection model."""

    @pytest.fixture
    def evaluation_data(self):
        """Sample evaluation data."""
        return {
            "id": "eval-1",
            "status": "completed",
            "status_description": "Done",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-1",
            "model_name": "Test Model",
            "model_key": "test-model",
            "model_company": "TestCorp",
            "dataset_id": "dataset-1",
            "dataset_name": "Test Dataset",
            "average_duration": 1000,
            "readability_score": 0.8,
            "toxicity_score": 0.1,
            "ethics_score": 0.9,
            "accuracy": 0.85,
        }

    def test_evaluations_with_list_of_evaluations(self, evaluation_data):
        """Evaluations model accepts list of Evaluation objects."""
        evaluations_data = {"data": [evaluation_data, evaluation_data]}
        evaluations = Evaluations(**evaluations_data)

        assert len(evaluations.data) == 2
        assert all(isinstance(eval, Evaluation) for eval in evaluations.data)
        assert evaluations.data[0].id == "eval-1"

    def test_evaluations_empty_list(self):
        """Evaluations model accepts empty list."""
        evaluations = Evaluations(data=[])

        assert evaluations.data == []
        assert isinstance(evaluations.data, list)

    def test_evaluations_invalid_data_structure(self):
        """Evaluations model validates data structure."""
        with pytest.raises(ValidationError):
            Evaluations(data="not-a-list")  # type: ignore[arg-type]


class TestResult:
    """Test Result model validation."""

    @pytest.fixture
    def valid_result_data(self):
        """Valid result data for testing."""
        return {
            "subset": "math",
            "prompt": "What is 2+2?",
            "result": "4",
            "truth": "4",
            "duration": timedelta(seconds=1.5),
            "score": 1.0,
            "metrics": {"accuracy": 1.0, "confidence": 0.95},
        }

    def test_result_creation(self, valid_result_data):
        """Result model creates with valid data."""
        result = Result(**valid_result_data)

        assert result.subset == "math"
        assert result.prompt == "What is 2+2?"
        assert result.score == 1.0
        assert isinstance(result.duration, timedelta)
        assert isinstance(result.metrics, dict)

    def test_result_timedelta_handling(self, valid_result_data):
        """Result model handles timedelta correctly."""
        result = Result(**valid_result_data)

        assert result.duration == timedelta(seconds=1.5)
        assert result.duration.total_seconds() == 1.5

    def test_result_metrics_validation(self, valid_result_data):
        """Result model validates metrics as dict."""
        result = Result(**valid_result_data)

        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["confidence"] == 0.95
        assert len(result.metrics) == 2

    def test_result_invalid_metrics_type(self, valid_result_data):
        """Result model rejects invalid metrics type."""
        invalid_data = valid_result_data.copy()
        invalid_data["metrics"] = "not-a-dict"

        with pytest.raises(ValidationError):
            Result(**invalid_data)


class TestResultMetrics:
    """Test ResultMetrics model."""

    @pytest.fixture
    def valid_metrics_data(self):
        """Valid result metrics data for testing."""
        return {
            "total_count": 150,
            "min_toxicity_score": 0.0,
            "max_toxicity_score": 0.8,
            "min_readability_score": 0.2,
            "max_readability_score": 0.95,
        }

    def test_result_metrics_creation(self, valid_metrics_data):
        """ResultMetrics model creates with valid data."""
        metrics = ResultMetrics(**valid_metrics_data)

        assert metrics.total_count == 150
        assert metrics.min_toxicity_score == 0.0
        assert metrics.max_toxicity_score == 0.8
        assert metrics.min_readability_score == 0.2
        assert metrics.max_readability_score == 0.95

    def test_result_metrics_optional_fields(self):
        """ResultMetrics model handles optional score fields."""
        metrics = ResultMetrics(
            total_count=100,
            min_toxicity_score=None,
            max_toxicity_score=None,
            min_readability_score=None,
            max_readability_score=None,
        )

        assert metrics.total_count == 100
        assert metrics.min_toxicity_score is None
        assert metrics.max_toxicity_score is None
        assert metrics.min_readability_score is None
        assert metrics.max_readability_score is None

    def test_result_metrics_field_types(self, valid_metrics_data):
        """ResultMetrics model enforces correct field types."""
        metrics = ResultMetrics(**valid_metrics_data)

        assert isinstance(metrics.total_count, int)
        assert isinstance(metrics.min_toxicity_score, (float, type(None)))
        assert isinstance(metrics.max_toxicity_score, (float, type(None)))
        assert isinstance(metrics.min_readability_score, (float, type(None)))
        assert isinstance(metrics.max_readability_score, (float, type(None)))

    def test_result_metrics_invalid_total_count(self, valid_metrics_data):
        """ResultMetrics model validates total_count as integer."""
        invalid_data = valid_metrics_data.copy()
        invalid_data["total_count"] = "not-an-int"

        with pytest.raises(ValidationError):
            ResultMetrics(**invalid_data)


class TestPagination:
    """Test Pagination model."""

    @pytest.fixture
    def valid_pagination_data(self):
        """Valid pagination data for testing."""
        return {
            "total_count": 250,
            "page_size": 100,
            "total_pages": 3,
        }

    def test_pagination_creation(self, valid_pagination_data):
        """Pagination model creates with valid data."""
        pagination = Pagination(**valid_pagination_data)

        assert pagination.total_count == 250
        assert pagination.page_size == 100
        assert pagination.total_pages == 3

    def test_pagination_field_types(self, valid_pagination_data):
        """Pagination model enforces correct field types."""
        pagination = Pagination(**valid_pagination_data)

        assert isinstance(pagination.total_count, int)
        assert isinstance(pagination.page_size, int)
        assert isinstance(pagination.total_pages, int)

    def test_pagination_zero_values(self):
        """Pagination model handles zero values correctly."""
        pagination = Pagination(
            total_count=0,
            page_size=100,
            total_pages=0,
        )

        assert pagination.total_count == 0
        assert pagination.page_size == 100
        assert pagination.total_pages == 0

    def test_pagination_validation_errors(self, valid_pagination_data):
        """Pagination model validates field types."""
        # Test invalid total_count
        invalid_data = valid_pagination_data.copy()
        invalid_data["total_count"] = "not-an-int"
        with pytest.raises(ValidationError):
            Pagination(**invalid_data)

        # Test invalid page_size
        invalid_data = valid_pagination_data.copy()
        invalid_data["page_size"] = 3.14
        with pytest.raises(ValidationError):
            Pagination(**invalid_data)

        # Test invalid total_pages
        invalid_data = valid_pagination_data.copy()
        invalid_data["total_pages"] = "not-an-int"
        with pytest.raises(ValidationError):
            Pagination(**invalid_data)


class TestResults:
    """Test Results collection model with pagination."""

    @pytest.fixture
    def valid_result_data(self):
        """Valid result data for testing."""
        return {
            "subset": "test",
            "prompt": "test prompt",
            "result": "test result",
            "truth": "test truth",
            "duration": timedelta(seconds=1),
            "score": 0.8,
            "metrics": {"score": 0.8},
        }

    @pytest.fixture
    def valid_metrics_data(self):
        """Valid metrics data for testing."""
        return {
            "total_count": 150,
            "min_toxicity_score": 0.0,
            "max_toxicity_score": 0.8,
            "min_readability_score": 0.2,
            "max_readability_score": 0.95,
        }

    @pytest.fixture
    def valid_pagination_data(self):
        """Valid pagination data for testing."""
        return {
            "total_count": 150,
            "page_size": 100,
            "total_pages": 2,
        }

    def test_results_with_pagination(self, valid_result_data, valid_metrics_data, valid_pagination_data):
        """Results model accepts all required fields including pagination."""
        results = Results(
            evaluation_id="eval-123",
            results=[valid_result_data, valid_result_data],
            metrics=valid_metrics_data,
            pagination=valid_pagination_data,
        )

        assert results.evaluation_id == "eval-123"
        assert len(results.results) == 2
        assert all(isinstance(result, Result) for result in results.results)
        assert isinstance(results.metrics, ResultMetrics)
        assert isinstance(results.pagination, Pagination)
        assert results.pagination.total_count == 150
        assert results.pagination.page_size == 100
        assert results.pagination.total_pages == 2

    def test_results_field_types(self, valid_result_data, valid_metrics_data, valid_pagination_data):
        """Results model enforces correct field types."""
        results = Results(
            evaluation_id="eval-456",
            results=[valid_result_data],
            metrics=valid_metrics_data,
            pagination=valid_pagination_data,
        )

        assert isinstance(results.evaluation_id, str)
        assert isinstance(results.results, list)
        assert isinstance(results.metrics, ResultMetrics)
        assert isinstance(results.pagination, Pagination)

    def test_results_empty_results_list(self, valid_metrics_data, valid_pagination_data):
        """Results model handles empty results list."""
        results = Results(
            evaluation_id="eval-empty",
            results=[],
            metrics=valid_metrics_data,
            pagination=valid_pagination_data,
        )

        assert results.evaluation_id == "eval-empty"
        assert len(results.results) == 0
        assert isinstance(results.results, list)
        assert isinstance(results.metrics, ResultMetrics)
        assert isinstance(results.pagination, Pagination)

    def test_results_validation_errors(self, valid_result_data, valid_metrics_data, valid_pagination_data):
        """Results model validates required fields."""
        # Test missing evaluation_id
        with pytest.raises(ValidationError):
            Results(
                results=[valid_result_data],
                metrics=valid_metrics_data,
                pagination=valid_pagination_data,
            )

        # Test missing metrics
        with pytest.raises(ValidationError):
            Results(
                evaluation_id="eval-123",
                results=[valid_result_data],
                pagination=valid_pagination_data,
            )

        # Test missing pagination
        with pytest.raises(ValidationError):
            Results(
                evaluation_id="eval-123",
                results=[valid_result_data],
                metrics=valid_metrics_data,
            )

    def test_results_nested_model_validation(self, valid_result_data, valid_pagination_data):
        """Results model validates nested models."""
        # Test invalid metrics
        with pytest.raises(ValidationError):
            Results(
                evaluation_id="eval-123",
                results=[valid_result_data],
                metrics="invalid-metrics",  # Should be ResultMetrics object
                pagination=valid_pagination_data,
            )

        # Test invalid pagination
        with pytest.raises(ValidationError):
            Results(
                evaluation_id="eval-123",
                results=[valid_result_data],
                metrics={
                    "total_count": 100,
                    "min_toxicity_score": 0.0,
                    "max_toxicity_score": 0.5,
                    "min_readability_score": 0.0,
                    "max_readability_score": 1.0,
                },
                pagination="invalid-pagination",  # Should be Pagination object
            )


class TestModel:
    """Test Model validation."""

    @pytest.fixture
    def valid_model_data(self):
        """Valid model data for testing."""
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

    def test_model_creation(self, valid_model_data):
        """Model creates with valid data."""
        model = Model(**valid_model_data)

        assert model.id == "model-123"
        assert model.name == "GPT-4"
        assert model.parameters == 1.76e12
        assert model.open_weights is False
        assert model.deprecated is False

    def test_model_boolean_fields(self, valid_model_data):
        """Model handles boolean fields correctly."""
        model = Model(**valid_model_data)

        assert isinstance(model.open_weights, bool)
        assert isinstance(model.deprecated, bool)
        assert model.open_weights is False

    def test_model_numeric_fields(self, valid_model_data):
        """Model validates numeric fields."""
        model = Model(**valid_model_data)

        assert isinstance(model.parameters, float)
        assert isinstance(model.context_length, int)
        assert isinstance(model.released_at, int)

    def test_model_field_validation(self, valid_model_data):
        """Model validates field types."""
        # Test numeric field validation
        invalid_data = valid_model_data.copy()
        invalid_data["parameters"] = "not-a-number"
        with pytest.raises(ValidationError):
            Model(**invalid_data)

        # Test int field validation
        invalid_data = valid_model_data.copy()
        invalid_data["context_length"] = "not-an-int"
        with pytest.raises(ValidationError):
            Model(**invalid_data)


class TestCustomModel:
    """Test CustomModel validation."""

    @pytest.fixture
    def valid_custom_model_data(self):
        """Valid custom model data."""
        return {
            "id": "custom-123",
            "key": "my-model",
            "name": "My Custom Model",
            "description": "Custom model description",
            "max_tokens": 4096,
            "api_url": "https://api.example.com/v1/chat",
            "disabled": False,
        }

    def test_custom_model_creation(self, valid_custom_model_data):
        """CustomModel creates with valid data."""
        model = CustomModel(**valid_custom_model_data)

        assert model.id == "custom-123"
        assert model.max_tokens == 4096
        assert model.api_url == "https://api.example.com/v1/chat"
        assert model.disabled is False

    def test_custom_model_url_validation(self, valid_custom_model_data):
        """CustomModel stores URL as string."""
        model = CustomModel(**valid_custom_model_data)

        assert isinstance(model.api_url, str)
        assert model.api_url.startswith("https://")


class TestModels:
    """Test Models collection with Union types."""

    def test_models_with_mixed_model_types(self):
        """Models collection handles Union of Model and CustomModel."""
        model_data = {
            "id": "model-1",
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

        custom_model_data = {
            "id": "custom-1",
            "key": "my-model",
            "name": "My Model",
            "description": "Custom",
            "max_tokens": 4096,
            "api_url": "https://api.example.com",
            "disabled": False,
        }

        models = Models(models=[model_data, custom_model_data])  # type: ignore[arg-type]

        assert len(models.models) == 2
        assert isinstance(models.models[0], Model)
        assert isinstance(models.models[1], CustomModel)


class TestBenchmark:
    """Test Benchmark model validation."""

    @pytest.fixture
    def valid_benchmark_data(self):
        """Valid benchmark data."""
        return {
            "id": "bench-123",
            "key": "mmlu",
            "name": "MMLU",
            "full_description": "Massive Multitask Language Understanding",
            "language": "english",
            "categories": ["reasoning", "knowledge"],
            "subsets": ["math", "science", "history"],
            "prompt_count": 15908,
            "deprecated": False,
        }

    def test_benchmark_creation(self, valid_benchmark_data):
        """Benchmark creates with valid data."""
        benchmark = Benchmark(**valid_benchmark_data)

        assert benchmark.id == "bench-123"
        assert benchmark.name == "MMLU"
        assert len(benchmark.categories) == 2
        assert len(benchmark.subsets) == 3
        assert benchmark.prompt_count == 15908

    def test_benchmark_list_fields(self, valid_benchmark_data):
        """Benchmark handles list fields correctly."""
        benchmark = Benchmark(**valid_benchmark_data)

        assert isinstance(benchmark.categories, list)
        assert isinstance(benchmark.subsets, list)
        assert "reasoning" in benchmark.categories
        assert "math" in benchmark.subsets


class TestCustomBenchmark:
    """Test CustomBenchmark with optional fields."""

    @pytest.fixture
    def valid_custom_benchmark_data(self):
        """Valid custom benchmark data."""
        return {
            "id": "custom-bench-123",
            "key": "my-benchmark",
            "name": "My Benchmark",
            "description": "Custom benchmark",
            "system_prompt": "You are a helpful assistant",
            "subsets": ["subset1", "subset2"],
            "prompt_count": 100,
            "version_count": 1,
            "regex_pattern": r"Answer: (.+)",
            "llm_judge_model_id": "gpt-4",
            "custom_instructions": "Rate on scale 1-10",
            "scoring_metric": "accuracy",
            "metrics": ["accuracy", "precision"],
            "files": ["data.jsonl"],
            "disabled": False,
        }

    def test_custom_benchmark_creation(self, valid_custom_benchmark_data):
        """CustomBenchmark creates with all fields."""
        benchmark = CustomBenchmark(**valid_custom_benchmark_data)

        assert benchmark.id == "custom-bench-123"
        assert benchmark.system_prompt == "You are a helpful assistant"
        assert benchmark.regex_pattern == r"Answer: (.+)"
        assert len(benchmark.metrics) == 2

    def test_custom_benchmark_optional_fields(self):
        """CustomBenchmark handles optional fields correctly."""
        minimal_data = {
            "id": "custom-123",
            "key": "test",
            "name": "Test",
            "description": "Test desc",
            "system_prompt": None,
            "subsets": ["test"],
            "prompt_count": 10,
            "version_count": 1,
            "regex_pattern": None,
            "llm_judge_model_id": "gpt-4",
            "custom_instructions": "Test",
            "scoring_metric": None,
            "metrics": ["accuracy"],
            "files": ["test.jsonl"],
            "disabled": False,
        }

        benchmark = CustomBenchmark(**minimal_data)

        assert benchmark.system_prompt is None
        assert benchmark.regex_pattern is None
        assert benchmark.scoring_metric is None


class TestBenchmarks:
    """Test Benchmarks collection with alias field."""

    def test_benchmarks_with_datasets_alias(self):
        """Benchmarks accepts 'datasets' as alias for benchmarks field."""
        benchmark_data = {
            "id": "bench-1",
            "key": "test",
            "name": "Test",
            "full_description": "Test benchmark",
            "language": "english",
            "categories": ["test"],
            "subsets": ["test"],
            "prompt_count": 10,
            "deprecated": False,
        }

        # Using the alias 'datasets'
        benchmarks = Benchmarks(datasets=[benchmark_data])  # type: ignore[arg-type]

        assert len(benchmarks.benchmarks) == 1
        assert isinstance(benchmarks.benchmarks[0], Benchmark)

    def test_benchmarks_field_validation(self):
        """Benchmarks validates field structure correctly."""
        # Should work with 'benchmarks' field name too
        benchmark_data = {
            "id": "bench-1",
            "key": "test",
            "name": "Test",
            "full_description": "Test benchmark",
            "language": "english",
            "categories": ["test"],
            "subsets": ["test"],
            "prompt_count": 10,
            "deprecated": False,
        }

        benchmarks = Benchmarks(datasets=[benchmark_data])  # type: ignore[arg-type]

        assert len(benchmarks.benchmarks) == 1


class TestModelSerialization:
    """Test model serialization and deserialization patterns."""

    def test_round_trip_serialization(self):
        """Models can be serialized and deserialized correctly."""
        original_data = {
            "id": "eval-123",
            "status": "completed",
            "status_description": "Done",
            "submitted_at": 1640995200,
            "finished_at": 1640995800,
            "model_id": "model-456",
            "model_name": "GPT-4",
            "model_key": "gpt-4",
            "model_company": "OpenAI",
            "dataset_id": "dataset-789",
            "dataset_name": "MMLU",
            "average_duration": 2500,
            "readability_score": 0.85,
            "toxicity_score": 0.02,
            "ethics_score": 0.92,
            "accuracy": 0.89,
        }

        # Create model, serialize, then deserialize
        evaluation = Evaluation(**original_data)
        serialized = evaluation.model_dump()
        deserialized = Evaluation(**serialized)

        assert deserialized.id == evaluation.id
        assert deserialized.accuracy == evaluation.accuracy
        assert deserialized == evaluation

    def test_json_compatibility(self):
        """Models work with JSON serialization."""
        import json

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

        model = Model(**model_data)
        json_str = json.dumps(model.model_dump())
        parsed_data = json.loads(json_str)
        reconstructed = Model(**parsed_data)

        assert reconstructed.name == model.name
        assert reconstructed.parameters == model.parameters
