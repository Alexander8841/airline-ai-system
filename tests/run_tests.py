import os
import sys

# Reconfigure stdout to use utf-8 to avoid CP1251 encode errors in Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tests
from tests.test_scenarios import (
    setup_function,
    test_database_initialization,
    test_search_flights_valid,
    test_search_flights_buggy_mode,
    test_get_booking_valid_and_invalid,
    test_change_booking_success,
    test_change_booking_class_mismatch,
    test_cancel_booking,
    test_get_policy
)

test_functions = [
    test_database_initialization,
    test_search_flights_valid,
    test_search_flights_buggy_mode,
    test_get_booking_valid_and_invalid,
    test_change_booking_success,
    test_change_booking_class_mismatch,
    test_cancel_booking,
    test_get_policy
]

passed = 0
failed = 0

print("=== Running Custom Test Runner ===")
for test in test_functions:
    try:
        setup_function()
        test()
        print(f"✅ {test.__name__}: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ {test.__name__}: FAILED")
        print(f"   Reason: {str(e)}")
        failed += 1

print("\n=== Test Session Summary ===")
print(f"Total: {len(test_functions)} | Passed: {passed} | Failed: {failed}")

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
