from unittest.mock import Mock, patch

import pytest

from atlas._resource import SyncAPIResource


class TestSyncAPIResource:
    """Test SyncAPIResource base class functionality."""

    @pytest.fixture
    def mock_client(self):
        """Mock Atlas client with required methods."""
        client = Mock()
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def resource_instance(self, mock_client):
        """Create SyncAPIResource instance for testing."""
        return SyncAPIResource(mock_client)

    def test_resource_initialization(self, mock_client):
        """SyncAPIResource initializes correctly with client."""
        resource = SyncAPIResource(mock_client)
        
        assert resource._client is mock_client
        assert resource._get is mock_client.get_cast
        assert resource._post is mock_client.post_cast

    def test_resource_stores_client_reference(self, resource_instance, mock_client):
        """Resource maintains reference to the client."""
        assert resource_instance._client is mock_client
        assert hasattr(resource_instance, '_client')

    def test_resource_delegates_get_to_client(self, resource_instance, mock_client):
        """_get method delegates to client.get_cast."""
        assert resource_instance._get is mock_client.get_cast
        
        # Verify it's the same method reference
        assert callable(resource_instance._get)
        
        # Test delegation works
        resource_instance._get("/test", params={"key": "value"})
        mock_client.get_cast.assert_called_once_with("/test", params={"key": "value"})

    def test_resource_delegates_post_to_client(self, resource_instance, mock_client):
        """_post method delegates to client.post_cast."""
        assert resource_instance._post is mock_client.post_cast
        
        # Verify it's the same method reference
        assert callable(resource_instance._post)
        
        # Test delegation works
        resource_instance._post("/test", body={"data": "test"})
        mock_client.post_cast.assert_called_once_with("/test", body={"data": "test"})

    def test_resource_sleep_method_exists(self, resource_instance):
        """Resource has _sleep method."""
        assert hasattr(resource_instance, '_sleep')
        assert callable(resource_instance._sleep)

    @patch('time.sleep')
    def test_resource_sleep_delegates_to_time_sleep(self, mock_time_sleep, resource_instance):
        """_sleep method delegates to time.sleep."""
        sleep_duration = 2.5
        
        resource_instance._sleep(sleep_duration)
        
        mock_time_sleep.assert_called_once_with(sleep_duration)

    @patch('time.sleep')
    def test_resource_sleep_with_different_durations(self, mock_time_sleep, resource_instance):
        """_sleep method works with various duration values."""
        durations = [0.1, 1.0, 5.0, 10.5, 60.0]
        
        for duration in durations:
            mock_time_sleep.reset_mock()
            resource_instance._sleep(duration)
            mock_time_sleep.assert_called_once_with(duration)

    @patch('time.sleep')
    def test_resource_sleep_with_zero_duration(self, mock_time_sleep, resource_instance):
        """_sleep method handles zero duration."""
        resource_instance._sleep(0.0)
        
        mock_time_sleep.assert_called_once_with(0.0)

    @patch('time.sleep')  
    def test_resource_sleep_with_integer_duration(self, mock_time_sleep, resource_instance):
        """_sleep method handles integer duration values."""
        resource_instance._sleep(3)
        
        mock_time_sleep.assert_called_once_with(3)

    def test_resource_initialization_with_different_clients(self):
        """SyncAPIResource works with different client objects."""
        # Test with different mock clients
        client1 = Mock()
        client1.get_cast = Mock(return_value="get_result_1")
        client1.post_cast = Mock(return_value="post_result_1")
        
        client2 = Mock() 
        client2.get_cast = Mock(return_value="get_result_2")
        client2.post_cast = Mock(return_value="post_result_2")
        
        resource1 = SyncAPIResource(client1)
        resource2 = SyncAPIResource(client2)
        
        # Verify each resource uses its own client
        assert resource1._client is client1
        assert resource2._client is client2
        assert resource1._get is client1.get_cast
        assert resource2._get is client2.get_cast
        
        # Verify method calls go to correct clients
        result1 = resource1._get("/test1")
        result2 = resource2._get("/test2")
        
        assert result1 == "get_result_1"
        assert result2 == "get_result_2"
        client1.get_cast.assert_called_once_with("/test1")
        client2.get_cast.assert_called_once_with("/test2")


class TestSyncAPIResourceInheritance:
    """Test SyncAPIResource as a base class for inheritance."""

    def test_resource_can_be_subclassed(self):
        """SyncAPIResource can be subclassed for specific resources."""
        
        class TestResource(SyncAPIResource):
            def get_data(self, id: str):
                return self._get(f"/data/{id}")
            
            def create_data(self, data: dict):
                return self._post("/data", body=data)
        
        mock_client = Mock()
        mock_client.get_cast = Mock(return_value={"id": "123", "data": "test"})
        mock_client.post_cast = Mock(return_value={"id": "456", "created": True})
        
        resource = TestResource(mock_client)
        
        # Test inherited initialization
        assert resource._client is mock_client
        assert resource._get is mock_client.get_cast
        assert resource._post is mock_client.post_cast
        
        # Test custom methods using inherited functionality
        get_result = resource.get_data("123")
        create_result = resource.create_data({"name": "test"})
        
        assert get_result == {"id": "123", "data": "test"}
        assert create_result == {"id": "456", "created": True}
        
        mock_client.get_cast.assert_called_once_with("/data/123")
        mock_client.post_cast.assert_called_once_with("/data", body={"name": "test"})

    def test_subclass_can_override_methods(self):
        """Subclasses can override resource methods."""
        
        class CustomResource(SyncAPIResource):
            def __init__(self, client):
                super().__init__(client)
                self.custom_property = "custom_value"
            
            def _sleep(self, seconds: float) -> None:
                # Custom sleep implementation
                self.last_sleep_duration = seconds
                super()._sleep(seconds)
        
        mock_client = Mock()
        mock_client.get_cast = Mock()
        mock_client.post_cast = Mock()
        
        resource = CustomResource(mock_client)
        
        # Test custom property
        assert resource.custom_property == "custom_value"
        
        # Test overridden method
        with patch('time.sleep') as mock_time_sleep:
            resource._sleep(1.5)
            
            assert resource.last_sleep_duration == 1.5
            mock_time_sleep.assert_called_once_with(1.5)

    def test_multiple_resource_instances_independent(self):
        """Multiple resource instances maintain independence."""
        
        class ResourceA(SyncAPIResource):
            def method_a(self):
                return self._get("/resource-a")
        
        class ResourceB(SyncAPIResource):
            def method_b(self):
                return self._post("/resource-b", body={"type": "b"})
        
        client1 = Mock()
        client1.get_cast = Mock(return_value="result_a")
        client1.post_cast = Mock()
        
        client2 = Mock()
        client2.get_cast = Mock()
        client2.post_cast = Mock(return_value="result_b")
        
        resource_a = ResourceA(client1)
        resource_b = ResourceB(client2)
        
        # Test that resources are independent
        result_a = resource_a.method_a()
        result_b = resource_b.method_b()
        
        assert result_a == "result_a"
        assert result_b == "result_b"
        
        # Verify correct clients were called
        client1.get_cast.assert_called_once_with("/resource-a")
        client2.post_cast.assert_called_once_with("/resource-b", body={"type": "b"})
        
        # Verify cross-contamination didn't occur
        client1.post_cast.assert_not_called()
        client2.get_cast.assert_not_called()


class TestSyncAPIResourceErrorHandling:
    """Test error handling in SyncAPIResource."""

    @pytest.fixture
    def mock_client(self):
        """Mock client that can raise errors."""
        client = Mock()
        client.get_cast = Mock()
        client.post_cast = Mock()
        return client

    @pytest.fixture
    def resource_instance(self, mock_client):
        """Create resource instance for error testing."""
        return SyncAPIResource(mock_client)

    def test_resource_propagates_get_errors(self, resource_instance, mock_client):
        """Resource propagates errors from _get calls."""
        from atlas._exceptions import APIStatusError
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        
        api_error = APIStatusError("Not Found", response=mock_response, body=None)
        mock_client.get_cast.side_effect = api_error
        
        with pytest.raises(APIStatusError):
            resource_instance._get("/test")

    def test_resource_propagates_post_errors(self, resource_instance, mock_client):
        """Resource propagates errors from _post calls."""
        from atlas._exceptions import APIConnectionError
        
        mock_request = Mock()
        connection_error = APIConnectionError(request=mock_request)
        mock_client.post_cast.side_effect = connection_error
        
        with pytest.raises(APIConnectionError):
            resource_instance._post("/test", body={"data": "test"})

    def test_resource_handles_client_method_missing(self):
        """Resource handles clients missing required methods gracefully."""
        # Create a client without the required methods
        incomplete_client = object()  # Plain object with no methods
        
        # This should fail during initialization since the methods don't exist
        with pytest.raises(AttributeError):
            SyncAPIResource(incomplete_client)  # type: ignore[arg-type]

    @patch('time.sleep')
    def test_resource_sleep_handles_exceptions(self, mock_time_sleep, resource_instance):
        """_sleep method handles exceptions from time.sleep."""
        mock_time_sleep.side_effect = KeyboardInterrupt("Interrupted")
        
        with pytest.raises(KeyboardInterrupt):
            resource_instance._sleep(1.0)
        
        mock_time_sleep.assert_called_once_with(1.0)


class TestSyncAPIResourceTyping:
    """Test type-related aspects of SyncAPIResource."""

    def test_resource_client_attribute_typing(self):
        """Resource._client maintains proper typing."""
        
        # Test with properly typed client (would be Atlas in real usage)
        mock_client = Mock()
        mock_client.get_cast = Mock()
        mock_client.post_cast = Mock()
        
        resource = SyncAPIResource(mock_client)
        
        # Verify the client is stored and accessible
        assert resource._client is mock_client
        assert hasattr(resource, '_client')

    def test_resource_method_signatures(self):
        """Resource methods have expected signatures."""
        import inspect
        
        # Check _sleep method signature
        sleep_sig = inspect.signature(SyncAPIResource._sleep)
        sleep_params = list(sleep_sig.parameters.keys())
        
        assert 'self' in sleep_params
        assert 'seconds' in sleep_params
        assert len(sleep_params) == 2

    def test_resource_initialization_signature(self):
        """Resource __init__ has expected signature."""
        import inspect
        
        init_sig = inspect.signature(SyncAPIResource.__init__)
        init_params = list(init_sig.parameters.keys())
        
        assert 'self' in init_params
        assert 'client' in init_params
        assert len(init_params) == 2


class TestSyncAPIResourceRealWorldUsage:
    """Test SyncAPIResource in realistic usage scenarios."""

    def test_resource_with_retry_logic(self):
        """Resource can implement retry logic using _sleep."""
        
        class RetryableResource(SyncAPIResource):
            def get_with_retry(self, url: str, max_retries: int = 3):
                for attempt in range(max_retries):
                    try:
                        return self._get(url)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        self._sleep(2 ** attempt)  # Exponential backoff
        
        mock_client = Mock()
        # First two calls fail, third succeeds
        mock_client.get_cast.side_effect = [
            Exception("First failure"),
            Exception("Second failure"), 
            {"success": True}
        ]
        
        resource = RetryableResource(mock_client)
        
        with patch.object(resource, '_sleep') as mock_sleep:
            result = resource.get_with_retry("/test")
            
            assert result == {"success": True}
            assert mock_client.get_cast.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)  # 2^0
            mock_sleep.assert_any_call(2)  # 2^1

    def test_resource_with_complex_workflow(self):
        """Resource can implement complex workflows."""
        
        class WorkflowResource(SyncAPIResource):
            def create_and_wait(self, data: dict, poll_interval: float = 1.0):
                # Create resource
                created = self._post("/create", body=data)
                resource_id = created["id"]  # type: ignore[index]
                
                # Poll until complete
                while True:
                    status = self._get(f"/status/{resource_id}")
                    if status["state"] == "completed":  # type: ignore[index]
                        return self._get(f"/result/{resource_id}")
                    elif status["state"] == "failed":  # type: ignore[index]
                        raise Exception("Workflow failed")
                    
                    self._sleep(poll_interval)
        
        mock_client = Mock()
        mock_client.post_cast.return_value = {"id": "workflow-123"}
        
        # Mock status progression: pending -> running -> completed
        mock_client.get_cast.side_effect = [
            {"state": "pending"},
            {"state": "running"},
            {"state": "completed"},
            {"result": "workflow complete"}
        ]
        
        resource = WorkflowResource(mock_client)
        
        with patch.object(resource, '_sleep') as mock_sleep:
            result = resource.create_and_wait({"name": "test"})
            
            assert result == {"result": "workflow complete"}
            assert mock_client.post_cast.call_count == 1
            assert mock_client.get_cast.call_count == 4
            assert mock_sleep.call_count == 2  # Two sleeps during polling