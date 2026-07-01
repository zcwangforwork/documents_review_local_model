#!/usr/bin/env python
"""Test vector_store.py add_documents batching logic"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent))

def test_batching_logic():
    """Test that add_documents correctly splits into 500-item batches"""
    # Create mock collection
    mock_collection = MagicMock()

    # Bypass ChromaDB initialization via __new__ (skip __init__ entirely)
    import vector_store
    # Inject a fake chromadb module to prevent import errors if __init__ is triggered
    fake_chromadb = MagicMock()
    sys.modules['chromadb'] = fake_chromadb
    sys.modules['chromadb.config'] = MagicMock()

    vs = vector_store.VectorStore.__new__(vector_store.VectorStore)
    vs.collection = mock_collection
    vs.persist_directory = "/tmp/test"
    vs.collection_name = "test_collection"
    vs.embedding_function = None

    # Test 1: Empty list
    print("Test 1: Empty document list")
    vs.add_documents([], [], [])
    assert mock_collection.add.call_count == 0, "Empty list should not call add"
    print("  PASSED")

    # Test 2: Single small batch (< 500)
    print("\nTest 2: Single small batch (10 documents)")
    mock_collection.reset_mock()
    docs = [f"doc_{i}" for i in range(10)]
    metas = [{"source": f"file_{i}"} for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    result = vs.add_documents(docs, metas, ids)
    assert mock_collection.add.call_count == 1, f"Expected 1 call, got {mock_collection.add.call_count}"
    assert len(result) == 10
    print(f"  PASSED: 1 batch for 10 docs")

    # Test 3: Exactly 500 documents (boundary)
    print("\nTest 3: Exactly 500 documents (boundary)")
    mock_collection.reset_mock()
    docs = [f"doc_{i}" for i in range(500)]
    metas = [{"source": f"file_{i}"} for i in range(500)]
    ids = [f"id_{i}" for i in range(500)]
    result = vs.add_documents(docs, metas, ids)
    assert mock_collection.add.call_count == 1, f"Expected 1 call, got {mock_collection.add.call_count}"
    assert len(result) == 500
    print(f"  PASSED: 1 batch for exactly 500 docs")

    # Test 4: 501 documents (should be 2 batches: 500 + 1)
    print("\nTest 4: 501 documents (2 batches: 500 + 1)")
    mock_collection.reset_mock()
    docs = [f"doc_{i}" for i in range(501)]
    metas = [{"source": f"file_{i}"} for i in range(501)]
    ids = [f"id_{i}" for i in range(501)]
    result = vs.add_documents(docs, metas, ids)
    assert mock_collection.add.call_count == 2, f"Expected 2 calls, got {mock_collection.add.call_count}"
    assert len(result) == 501

    # Verify first call had 500, second had 1
    call1_docs = mock_collection.add.call_args_list[0][1]['documents']
    call2_docs = mock_collection.add.call_args_list[1][1]['documents']
    assert len(call1_docs) == 500, f"First batch should be 500, got {len(call1_docs)}"
    assert len(call2_docs) == 1, f"Second batch should be 1, got {len(call2_docs)}"
    print(f"  PASSED: Batch 1={len(call1_docs)}, Batch 2={len(call2_docs)}")

    # Test 5: 1500 documents (3 batches: 500 + 500 + 500)
    print("\nTest 5: 1500 documents (3 batches)")
    mock_collection.reset_mock()
    docs = [f"doc_{i}" for i in range(1500)]
    metas = [{"source": f"file_{i}"} for i in range(1500)]
    ids = [f"id_{i}" for i in range(1500)]
    result = vs.add_documents(docs, metas, ids)
    assert mock_collection.add.call_count == 3, f"Expected 3 calls, got {mock_collection.add.call_count}"
    assert len(result) == 1500

    for i, call_args in enumerate(mock_collection.add.call_args_list):
        batch_size = len(call_args[1]['documents'])
        assert batch_size == 500, f"Batch {i+1} should be 500, got {batch_size}"
    print(f"  PASSED: 3 batches of 500 each")

    # Test 6: With embeddings (should also batch correctly)
    print("\nTest 6: 800 documents with embeddings (2 batches)")
    mock_collection.reset_mock()
    docs = [f"doc_{i}" for i in range(800)]
    metas = [{"source": f"file_{i}"} for i in range(800)]
    ids = [f"id_{i}" for i in range(800)]
    embs = [[float(i)] * 1024 for i in range(800)]
    result = vs.add_documents(docs, metas, ids, embeddings=embs)
    assert mock_collection.add.call_count == 2, f"Expected 2 calls, got {mock_collection.add.call_count}"

    # Verify embeddings were passed in batches
    call1 = mock_collection.add.call_args_list[0][1]
    call2 = mock_collection.add.call_args_list[1][1]
    assert 'embeddings' in call1, "First call should have embeddings"
    assert 'embeddings' in call2, "Second call should have embeddings"
    assert len(call1['embeddings']) == 500
    assert len(call2['embeddings']) == 300
    print(f"  PASSED: Embeddings batched correctly (500 + 300)")

    print("\n" + "=" * 60)
    print("ALL vector_store batching tests PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_batching_logic()