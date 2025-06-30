#!/usr/bin/env python3
"""
End-to-end test demonstrating vector security validation fixes.

This script tests all the critical vector fields that were previously vulnerable
to DoS attacks through unlimited vector dimensions.
"""

import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vector_security():
    """Test all vector security validators are working correctly."""
    
    # Import all the models with vector fields
    from src.models.vector_search import (
        SearchStage,
        BasicFilteredSearchRequest,
        BasicHybridSearchRequest,
        SearchResult,
        FilteredSearchRequest,
    )
    
    print("🔒 Testing Vector Security Validation Fixes")
    print("=" * 60)
    
    # Valid test vector
    valid_vector = [1.0, 2.0, 3.0]
    
    # Test each model class with valid vectors first
    print("\n✅ Testing valid vectors...")
    
    # 1. SearchStage
    stage = SearchStage(
        query_vector=valid_vector,
        vector_name="dense",
        vector_type="DENSE",
        limit=10
    )
    print(f"SearchStage: {len(stage.query_vector)} dimensions")
    
    # 2. BasicFilteredSearchRequest
    filtered_req = BasicFilteredSearchRequest(
        collection_name="test",
        query_vector=valid_vector,
        filters={"type": "document"}
    )
    print(f"BasicFilteredSearchRequest: {len(filtered_req.query_vector)} dimensions")
    
    # 3. BasicHybridSearchRequest
    hybrid_req = BasicHybridSearchRequest(
        collection_name="test",
        dense_vector=valid_vector
    )
    print(f"BasicHybridSearchRequest: {len(hybrid_req.dense_vector)} dimensions")
    
    # 4. SearchResult (vector is optional)
    result = SearchResult(
        id="doc1",
        score=0.95,
        vector=valid_vector
    )
    print(f"SearchResult: {len(result.vector) if result.vector else 0} dimensions")
    
    # 5. FilteredSearchRequest
    filtered_adv_req = FilteredSearchRequest(
        collection_name="test",
        query_vector=valid_vector
    )
    print(f"FilteredSearchRequest: {len(filtered_adv_req.query_vector)} dimensions")
    
    # Test DoS protection
    print("\n🛡️ Testing DoS protection (unlimited vector dimensions)...")
    
    test_cases = [
        ("Empty vector", []),
        ("Oversized vector (5000 dims)", [1.0] * 5000),
        ("Massive DoS vector (100k dims)", [1.0] * 100000),
    ]
    
    models_to_test = [
        ("SearchStage", lambda v: SearchStage(query_vector=v, vector_name="dense", vector_type="DENSE", limit=10)),
        ("BasicFilteredSearchRequest", lambda v: BasicFilteredSearchRequest(collection_name="test", query_vector=v, filters={})),
        ("BasicHybridSearchRequest", lambda v: BasicHybridSearchRequest(collection_name="test", dense_vector=v)),
        ("SearchResult", lambda v: SearchResult(id="test", score=0.5, vector=v)),
        ("FilteredSearchRequest", lambda v: FilteredSearchRequest(collection_name="test", query_vector=v)),
    ]
    
    dos_blocked = 0
    total_dos_tests = len(test_cases) * len(models_to_test)
    
    for test_name, test_vector in test_cases:
        for model_name, model_constructor in models_to_test:
            try:
                model_constructor(test_vector)
                print(f"❌ {model_name} with {test_name}: Should have been blocked!")
            except ValueError as e:
                if "security: DoS prevention" in str(e):
                    dos_blocked += 1
                    print(f"✅ {model_name} blocked {test_name}")
                else:
                    print(f"⚠️ {model_name} blocked {test_name} but wrong error: {e}")
            except Exception as e:
                print(f"⚠️ {model_name} with {test_name}: Unexpected error: {e}")
    
    # Test data integrity protection
    print("\n🛡️ Testing data integrity protection (NaN/Inf/invalid values)...")
    
    integrity_test_cases = [
        ("NaN injection", [1.0, float('nan'), 3.0]),
        ("Positive infinity", [1.0, float('inf'), 3.0]),
        ("Negative infinity", [1.0, float('-inf'), 3.0]),
        ("String injection", [1.0, "malicious", 3.0]),
        ("None injection", [1.0, None, 3.0]),
        ("Dict injection", [1.0, {"attack": "payload"}, 3.0]),
    ]
    
    integrity_blocked = 0
    total_integrity_tests = len(integrity_test_cases) * len(models_to_test)
    
    for test_name, test_vector in integrity_test_cases:
        for model_name, model_constructor in models_to_test:
            try:
                model_constructor(test_vector)
                print(f"❌ {model_name} with {test_name}: Should have been blocked!")
            except (ValueError, TypeError) as e:
                if any(keyword in str(e) for keyword in ["security:", "numeric", "NaN", "Inf"]):
                    integrity_blocked += 1
                    print(f"✅ {model_name} blocked {test_name}")
                else:
                    print(f"⚠️ {model_name} blocked {test_name} but wrong error: {e}")
            except Exception as e:
                print(f"⚠️ {model_name} with {test_name}: Unexpected error: {e}")
    
    # Test edge cases
    print("\n🎯 Testing edge cases...")
    
    # Maximum allowed vector (4096 dimensions)
    max_vector = [1.0] * 4096
    stage_max = SearchStage(
        query_vector=max_vector,
        vector_name="dense",
        vector_type="DENSE",
        limit=10
    )
    print(f"✅ Maximum vector size accepted: {len(stage_max.query_vector)} dimensions")
    
    # Optional vector field (SearchResult.vector can be None)
    result_no_vector = SearchResult(id="doc2", score=0.8, vector=None)
    print(f"✅ Optional vector field works: {result_no_vector.vector is None}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🔒 VECTOR SECURITY VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Valid vectors: All 5 model classes working correctly")
    print(f"🛡️ DoS protection: {dos_blocked}/{total_dos_tests} attacks blocked")
    print(f"🛡️ Data integrity: {integrity_blocked}/{total_integrity_tests} attacks blocked")
    print(f"🎯 Edge cases: Maximum dimensions (4096) accepted, None vectors handled")
    
    if dos_blocked == total_dos_tests and integrity_blocked == total_integrity_tests:
        print(f"\n🎉 SUCCESS: All security validations working correctly!")
        return True
    else:
        print(f"\n❌ FAILURE: Some security validations failed!")
        return False

if __name__ == "__main__":
    success = test_vector_security()
    sys.exit(0 if success else 1)