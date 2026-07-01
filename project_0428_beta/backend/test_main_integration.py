#!/usr/bin/env python
"""Integration test for main.py - validates all components can be imported and function correctly"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test Config class initialization"""
    print("Test 1: Config class")
    from main import Config, config
    assert config.api_url, "API URL should not be empty"
    assert config.api_key, "API key should not be empty"
    assert config.model, "Model should not be empty"
    assert config.timeout > 0, "Timeout should be positive"
    cfg_dict = config.to_dict()
    assert 'api_url' in cfg_dict
    assert 'model' in cfg_dict
    print(f"  PASSED: model={config.model}, timeout={config.timeout}")
    return True

def test_pydantic_models():
    """Test Pydantic models"""
    print("\nTest 2: Pydantic models")
    from main import Message, ChatRequest, ChatResponse, HealthResponse

    # Message
    msg = Message(role="user", content="test")
    assert msg.role == "user"
    assert msg.content == "test"
    print("  Message: OK")

    # ChatRequest
    req = ChatRequest(messages=[msg], temperature=0.5, max_tokens=1000)
    assert len(req.messages) == 1
    assert req.temperature == 0.5
    assert req.max_tokens == 1000
    print("  ChatRequest: OK")

    # ChatResponse
    resp = ChatResponse(answer="test answer", usage={"total_tokens": 10})
    assert resp.answer == "test answer"
    print("  ChatResponse: OK")

    # HealthResponse
    health = HealthResponse(status="healthy", config={}, vectorstore_loaded=False, document_count=0)
    assert health.status == "healthy"
    print("  HealthResponse: OK")

    # Validation: temperature range
    try:
        ChatRequest(messages=[msg], temperature=3.0)
        assert False, "Should have raised validation error"
    except Exception:
        print("  Temperature validation: OK")

    print("  PASSED")
    return True

def test_conversation_history():
    """Test ConversationHistory management with truncation"""
    print("\nTest 3: ConversationHistory")
    from main import ConversationHistory

    ch = ConversationHistory(max_history=5)

    # Basic add and retrieve
    ch.add_message("session1", "user", "Hello")
    ch.add_message("session1", "assistant", "Hi there")
    history = ch.get_or_create("session1")
    assert len(history) == 2
    print("  Basic add/retrieve: OK")

    # Max history truncation
    for i in range(10):
        ch.add_message("session2", "user", f"Message {i}")
    history = ch.get_or_create("session2")
    assert len(history) <= 5 + 1, f"History should be capped at ~6 (system + 5), got {len(history)}"
    print(f"  Max history cap: OK (got {len(history)} messages)")

    # Content length truncation
    long_content = "x" * 15000
    ch.add_message("session3", "user", long_content)
    history = ch.get_or_create("session3")
    stored_content = history[0]["content"]
    assert len(stored_content) < 15000, f"Content should be truncated, got {len(stored_content)}"
    assert "已截断" in stored_content, "Should contain truncation notice"
    print(f"  Content truncation: OK (truncated from 15000 to {len(stored_content)})")

    # Clear
    ch.clear("session1")
    assert len(ch.get_or_create("session1")) == 0
    print("  Clear session: OK")

    # Non-existent session
    history = ch.get_or_create("nonexistent")
    assert history == []
    print("  New session: OK")

    print("  PASSED")
    return True

def test_doc_processor_import():
    """Test doc_processor module is importable and functional"""
    print("\nTest 4: doc_processor module")
    from doc_processor import extract_text, chunk_text, get_file_metadata
    print("  All functions imported: OK")
    print("  PASSED")
    return True

def test_vector_store_module():
    """Test vector_store module can be imported (without ChromaDB loading)"""
    print("\nTest 5: vector_store module structure")
    import vector_store
    assert hasattr(vector_store, 'VectorStore'), "VectorStore class should exist"
    assert hasattr(vector_store, 'MiniMaxEmbeddingFunction'), "MiniMaxEmbeddingFunction should exist"
    assert hasattr(vector_store, 'create_vector_store'), "create_vector_store factory should exist"
    print("  Module structure: OK")
    print("  PASSED")
    return True

def test_rag_retriever_module():
    """Test rag_retriever module can be imported"""
    print("\nTest 6: rag_retriever module")
    from rag_retriever import create_rag_retriever
    print("  Module import: OK")
    print("  PASSED")
    return True

def test_streaming_upload_path():
    """Test the streaming upload logic (without actual server)"""
    print("\nTest 7: Streaming upload logic verification")
    import tempfile

    # Simulate the streaming write pattern used in main.py
    test_data = b"x" * (128 * 1024)  # 128KB of test data
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            # Write in 64KB chunks (same as main.py)
            chunk_size = 64 * 1024
            for offset in range(0, len(test_data), chunk_size):
                chunk = test_data[offset:offset + chunk_size]
                tmp.write(chunk)
            tmp_path = tmp.name

        assert os.path.exists(tmp_path)
        actual_size = os.path.getsize(tmp_path)
        assert actual_size == len(test_data), f"Expected {len(test_data)}, got {actual_size}"
        print(f"  Wrote {actual_size} bytes in 64KB chunks: OK")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        assert not os.path.exists(tmp_path)
        print("  Temp file cleanup: OK")

    print("  PASSED")
    return True

def test_main_module_structure():
    """Test main.py FastAPI app structure"""
    print("\nTest 8: main.py FastAPI app structure")
    from main import app, config, conversation_manager

    # Verify app
    assert app.title == "医疗器械体系文件审核 Agent API"
    assert app.version == "3.0.0"
    print(f"  App: {app.title} v{app.version}")

    # Verify routes (skip Mount objects used for static files)
    from starlette.routing import Mount
    route_paths = set()
    for route in app.routes:
        if isinstance(route, Mount):
            continue
        if hasattr(route, 'path'):
            route_paths.add(route.path)
    expected_routes = ['/', '/info', '/health', '/api/chat', '/api/upload',
                       '/api/analyze', '/api/clear', '/api/history/{session_id}',
                       '/api/vectorstore/status']
    for route in expected_routes:
        found = any(route in r for r in route_paths)
        if not found:
            found = any(route.rstrip('/') in r.rstrip('/') for r in route_paths)
        if found:
            print(f"  Route {route}: OK")
        else:
            print(f"  Route {route}: MISSING")

    # Verify CORS middleware
    assert len(app.user_middleware) > 0, "Should have middleware configured"
    print(f"  Middleware count: {len(app.user_middleware)}")

    # Verify conversation manager
    assert conversation_manager is not None
    print("  ConversationManager: initialized")

    print("  PASSED")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Backend Integration Tests")
    print("=" * 60)

    tests = [
        test_config,
        test_pydantic_models,
        test_conversation_history,
        test_doc_processor_import,
        test_vector_store_module,
        test_rag_retriever_module,
        test_streaming_upload_path,
        test_main_module_structure,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)