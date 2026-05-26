from unittest.mock import Mock

import pytest

from layerlens.models import (
    Evaluation,
    Pagination,
    EvaluationStatus,
    PublicModelDetail,
    ComparisonResponse,
    EvaluationsResponse,
    PublicBenchmarkDetail,
    PublicModelsListResponse,
    PublicBenchmarksListResponse,
)
from layerlens.resources.comparisons.comparisons import Comparisons


def _make_eval(eval_id: str, model_id: str, benchmark_id: str) -> Evaluation:
    return Evaluation(
        id=eval_id,
        status=EvaluationStatus.SUCCESS,
        submitted_at=1640995200,
        finished_at=1640995800,
        model_id=model_id,
        dataset_id=benchmark_id,
        average_duration=2500,
        accuracy=0.89,
    )


def _make_eval_response(evaluations: list[Evaluation]) -> EvaluationsResponse:
    return EvaluationsResponse(
        evaluations=evaluations,
        pagination=Pagination(
            page=1,
            page_size=1,
            total_pages=1,
            total_count=len(evaluations),
        ),
    )


def _make_models_response(models: list[PublicModelDetail]) -> PublicModelsListResponse:
    return PublicModelsListResponse(
        models=models,
        count=len(models),
        total_count=len(models),
    )


def _make_benchmarks_response(benchmarks: list[PublicBenchmarkDetail]) -> PublicBenchmarksListResponse:
    return PublicBenchmarksListResponse(
        datasets=benchmarks,
        count=len(benchmarks),
        total_count=len(benchmarks),
    )


class TestCompareModels:
    """Test Comparisons.compare_models convenience method."""

    @pytest.fixture
    def mock_public_client(self):
        client = Mock()
        client.get_cast = Mock()
        client.evaluations = Mock()
        client.models = Mock()
        client.benchmarks = Mock()
        return client

    @pytest.fixture
    def comparisons(self, mock_public_client):
        return Comparisons(mock_public_client)

    def test_compare_models_success(self, comparisons, mock_public_client):
        """compare_models finds evaluations for both models and calls compare."""
        eval1 = _make_eval("eval-1", "model-a", "bench-1")
        eval2 = _make_eval("eval-2", "model-b", "bench-1")

        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([eval2]),
        ]

        comparisons._get.return_value = {
            "results": [],
            "total_count": 0,
            "correct_count_1": 5,
            "total_results_1": 10,
            "correct_count_2": 7,
            "total_results_2": 10,
        }

        result = comparisons.compare_models(
            benchmark_id="bench-1",
            model_id_1="model-a",
            model_id_2="model-b",
        )

        assert isinstance(result, ComparisonResponse)

        # Verify get_many was called correctly for both models
        calls = mock_public_client.evaluations.get_many.call_args_list
        assert len(calls) == 2

        assert calls[0].kwargs["model_ids"] == ["model-a"]
        assert calls[0].kwargs["benchmark_ids"] == ["bench-1"]
        assert calls[0].kwargs["status"] == EvaluationStatus.SUCCESS
        assert calls[0].kwargs["sort_by"] == "submitted_at"
        assert calls[0].kwargs["order"] == "desc"
        assert calls[0].kwargs["page_size"] == 1
        assert calls[0].kwargs["unique"] is True

        assert calls[1].kwargs["model_ids"] == ["model-b"]

        # Verify compare was called with the found evaluation IDs
        compare_call = comparisons._get.call_args
        params = compare_call.kwargs.get("params") or compare_call[1].get("params")
        assert params["evaluation_id_1"] == "eval-1"
        assert params["evaluation_id_2"] == "eval-2"

    def test_compare_models_model_1_not_found(self, comparisons, mock_public_client):
        """compare_models raises ValueError when model 1 has no evaluation."""
        mock_public_client.evaluations.get_many.return_value = _make_eval_response([])

        with pytest.raises(ValueError, match="model-a"):
            comparisons.compare_models(
                benchmark_id="bench-1",
                model_id_1="model-a",
                model_id_2="model-b",
            )

    def test_compare_models_model_2_not_found(self, comparisons, mock_public_client):
        """compare_models raises ValueError when model 2 has no evaluation."""
        eval1 = _make_eval("eval-1", "model-a", "bench-1")

        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([]),
        ]

        with pytest.raises(ValueError, match="model-b"):
            comparisons.compare_models(
                benchmark_id="bench-1",
                model_id_1="model-a",
                model_id_2="model-b",
            )

    def test_compare_models_none_response(self, comparisons, mock_public_client):
        """compare_models raises ValueError when get_many returns None."""
        mock_public_client.evaluations.get_many.return_value = None

        with pytest.raises(ValueError, match="model-a"):
            comparisons.compare_models(
                benchmark_id="bench-1",
                model_id_1="model-a",
                model_id_2="model-b",
            )

    def test_compare_models_passes_through_params(self, comparisons, mock_public_client):
        """compare_models forwards pagination, filter, and search to compare."""
        eval1 = _make_eval("eval-1", "model-a", "bench-1")
        eval2 = _make_eval("eval-2", "model-b", "bench-1")

        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([eval2]),
        ]
        comparisons._get.return_value = {
            "results": [],
            "total_count": 0,
            "correct_count_1": 0,
            "total_results_1": 0,
            "correct_count_2": 0,
            "total_results_2": 0,
        }

        comparisons.compare_models(
            benchmark_id="bench-1",
            model_id_1="model-a",
            model_id_2="model-b",
            page=2,
            page_size=50,
            outcome_filter="both_succeed",
            search="test query",
        )

        compare_call = comparisons._get.call_args
        params = compare_call.kwargs.get("params") or compare_call[1].get("params")
        assert params["page"] == "2"
        assert params["page_size"] == "50"
        assert params["outcome_filter"] == "both_succeed"
        assert params["search"] == "test query"

    def test_compare_models_picks_most_recent(self, comparisons, mock_public_client):
        """compare_models requests sort by submittedAt desc to get the most recent."""
        eval1 = _make_eval("eval-1", "model-a", "bench-1")
        eval2 = _make_eval("eval-2", "model-b", "bench-1")

        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([eval2]),
        ]
        comparisons._get.return_value = {
            "results": [],
            "total_count": 0,
            "correct_count_1": 0,
            "total_results_1": 0,
            "correct_count_2": 0,
            "total_results_2": 0,
        }

        comparisons.compare_models(
            benchmark_id="bench-1",
            model_id_1="model-a",
            model_id_2="model-b",
        )

        for call in mock_public_client.evaluations.get_many.call_args_list:
            assert call.kwargs["sort_by"] == "submitted_at"
            assert call.kwargs["order"] == "desc"
            assert call.kwargs["page_size"] == 1
            assert call.kwargs["status"] == EvaluationStatus.SUCCESS
            assert call.kwargs["unique"] is True

    def test_compare_models_resolves_keys(self, comparisons, mock_public_client):
        """compare_models resolves benchmark_key/model_key_* into IDs before lookup."""
        mock_public_client.benchmarks.get.return_value = _make_benchmarks_response(
            [PublicBenchmarkDetail(id="bench-1", key="mmlu_pro", name="MMLU Pro")]
        )
        mock_public_client.models.get.side_effect = [
            _make_models_response([PublicModelDetail(id="model-a", key="gpt-4", name="GPT-4")]),
            _make_models_response([PublicModelDetail(id="model-b", key="claude-opus", name="Claude Opus")]),
        ]

        eval1 = _make_eval("eval-1", "model-a", "bench-1")
        eval2 = _make_eval("eval-2", "model-b", "bench-1")
        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([eval2]),
        ]
        comparisons._get.return_value = {
            "results": [],
            "total_count": 0,
            "correct_count_1": 0,
            "total_results_1": 0,
            "correct_count_2": 0,
            "total_results_2": 0,
        }

        result = comparisons.compare_models(
            benchmark_key="mmlu_pro",
            model_key_1="gpt-4",
            model_key_2="claude-opus",
        )

        assert isinstance(result, ComparisonResponse)

        # Benchmark + model keys were each looked up
        mock_public_client.benchmarks.get.assert_called_once()
        assert mock_public_client.benchmarks.get.call_args.kwargs["key"] == "mmlu_pro"

        model_get_keys = [c.kwargs["key"] for c in mock_public_client.models.get.call_args_list]
        assert model_get_keys == ["gpt-4", "claude-opus"]

        # Resolved IDs are forwarded to evaluations.get_many
        eval_calls = mock_public_client.evaluations.get_many.call_args_list
        assert eval_calls[0].kwargs["model_ids"] == ["model-a"]
        assert eval_calls[0].kwargs["benchmark_ids"] == ["bench-1"]
        assert eval_calls[1].kwargs["model_ids"] == ["model-b"]

    def test_compare_models_mixed_id_and_key(self, comparisons, mock_public_client):
        """compare_models accepts mixing IDs for some entities and keys for others."""
        mock_public_client.models.get.return_value = _make_models_response(
            [PublicModelDetail(id="model-b", key="claude-opus", name="Claude Opus")]
        )

        eval1 = _make_eval("eval-1", "model-a", "bench-1")
        eval2 = _make_eval("eval-2", "model-b", "bench-1")
        mock_public_client.evaluations.get_many.side_effect = [
            _make_eval_response([eval1]),
            _make_eval_response([eval2]),
        ]
        comparisons._get.return_value = {
            "results": [],
            "total_count": 0,
            "correct_count_1": 0,
            "total_results_1": 0,
            "correct_count_2": 0,
            "total_results_2": 0,
        }

        comparisons.compare_models(
            benchmark_id="bench-1",
            model_id_1="model-a",
            model_key_2="claude-opus",
        )

        mock_public_client.benchmarks.get.assert_not_called()
        mock_public_client.models.get.assert_called_once()
        assert mock_public_client.models.get.call_args.kwargs["key"] == "claude-opus"

        eval_calls = mock_public_client.evaluations.get_many.call_args_list
        assert eval_calls[1].kwargs["model_ids"] == ["model-b"]

    def test_compare_models_rejects_both_id_and_key(self, comparisons):
        """Supplying both ID and key for the same entity is an error."""
        with pytest.raises(ValueError, match="benchmark_id"):
            comparisons.compare_models(
                benchmark_id="bench-1",
                benchmark_key="mmlu_pro",
                model_id_1="model-a",
                model_id_2="model-b",
            )

    def test_compare_models_rejects_neither_id_nor_key(self, comparisons):
        """Supplying neither ID nor key for an entity is an error."""
        with pytest.raises(ValueError, match="model_id_1"):
            comparisons.compare_models(
                benchmark_id="bench-1",
                model_id_2="model-b",
            )

    def test_compare_models_unknown_key_raises(self, comparisons, mock_public_client):
        """An unresolvable key raises ValueError with the key in the message."""
        mock_public_client.benchmarks.get.return_value = _make_benchmarks_response([])

        with pytest.raises(ValueError, match="missing-bench"):
            comparisons.compare_models(
                benchmark_key="missing-bench",
                model_id_1="model-a",
                model_id_2="model-b",
            )
