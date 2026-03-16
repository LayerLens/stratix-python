from unittest.mock import Mock

import pytest

from layerlens.models import (
    Evaluation,
    Pagination,
    EvaluationStatus,
    ComparisonResponse,
    EvaluationsResponse,
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


class TestCompareModels:
    """Test Comparisons.compare_models convenience method."""

    @pytest.fixture
    def mock_public_client(self):
        client = Mock()
        client.get_cast = Mock()
        client.evaluations = Mock()
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
