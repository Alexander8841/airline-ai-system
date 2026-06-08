import os
import json
from src.database import DB, reset_db, snapshot_db
from src.tools import (
    get_booking, 
    search_flights, 
    change_booking, 
    cancel_booking, 
    get_policy,
    update_booking
)

def setup_function():
    """Ensure a clean database state before each test scenario."""
    reset_db()

def test_database_initialization():
    """Verify that flights, bookings, and policies load correctly."""
    assert len(DB["flights"]) == 8
    assert len(DB["bookings"]) == 2
    assert "rebooking" in DB["policies"]["details"]
    assert "cancellation" in DB["policies"]["details"]

def test_search_flights_valid():
    """Verify flight search yields correct results for a valid route."""
    # Search flights MOW -> PAR on Feb 20, 2026
    # (should find SU2454_0220, SU2456_0220, SU2458_0220, AF1845_0220)
    result_json = search_flights.invoke({"origin": "MOW", "destination": "PAR", "date": "2026-02-20"})
    results = json.loads(result_json)
    
    assert len(results) == 4
    for f in results:
        assert f["from"] == "MOW"
        assert f["to"] == "PAR"
        assert f["date"] == "2026-02-20"
        assert "flight_key" in f

def test_search_flights_buggy_mode():
    """Verify that search_flights hides flight_key when BUGGY_TOOLS is enabled."""
    os.environ["BUGGY_TOOLS"] = "true"
    try:
        result_json = search_flights.invoke({"origin": "MOW", "destination": "PAR", "date": "2026-02-20"})
        results = json.loads(result_json)
        
        assert len(results) == 4
        for f in results:
            assert "flight_key" not in f
            assert "id" in f
    finally:
        # Cleanup
        os.environ["BUGGY_TOOLS"] = "false"

def test_get_booking_valid_and_invalid():
    """Verify booking lookup yields correct JSON and errors on invalid inputs."""
    # Valid lookup
    valid_res = json.loads(get_booking.invoke({"booking_id": "ABC123"}))
    assert valid_res["passenger"] == "Ivan Petrov"
    assert valid_res["booking_id"] == "ABC123"
    assert valid_res["flight"]["id"] == "SU2454"
    
    # Invalid lookup
    invalid_res = json.loads(get_booking.invoke({"booking_id": "FAKE000"}))
    assert "error" in invalid_res

def test_change_booking_success():
    """Verify booking change mutates status, updates flight keys, and seats counts."""
    # Move Ivan Petrov (ABC123, currently economy on SU2454_0220 with 5 seats) 
    # to SU2456_0220 (MOW->PAR on Feb 20, economy with 25 seats)
    
    # Check pre-conditions
    old_seats = DB["flights"]["SU2454_0220"]["seats"]  # 5
    new_seats = DB["flights"]["SU2456_0220"]["seats"]  # 25
    
    change_res = json.loads(change_booking.invoke({
        "booking_id": "ABC123",
        "new_flight_key": "SU2456_0220"
    }))
    
    assert change_res["success"] is True
    assert change_res["new_flight"] == "SU2456"
    assert change_res["price_diff"] == 2000 # 27000 - 25000
    
    # Post-conditions checks
    assert DB["bookings"]["ABC123"]["flight_key"] == "SU2456_0220"
    assert DB["bookings"]["ABC123"]["status"] == "changed"
    assert DB["flights"]["SU2454_0220"]["seats"] == old_seats + 1  # returned 1 seat
    assert DB["flights"]["SU2456_0220"]["seats"] == new_seats - 1  # took 1 seat

def test_change_booking_class_mismatch():
    """Verify class policies. (update_booking checks class mismatch)"""
    # Try updating booking XYZ789 (Business) to SU2454 (Economy)
    # this is checked via update_booking tool from lecture 3
    res = json.loads(update_booking.invoke({
        "booking_id": "XYZ789",
        "new_flight_number": "SU2454",
        "new_date": "2026-02-20"
    }))
    assert res["status"] == "error"
    assert "Class mismatch" in res["message"]

def test_cancel_booking():
    """Verify cancellation refunds seats and changes booking status."""
    old_seats = DB["flights"]["SU2454_0220"]["seats"]  # 5
    
    cancel_res = json.loads(cancel_booking.invoke({"booking_id": "ABC123"}))
    
    assert cancel_res["success"] is True
    assert cancel_res["refund"] == 25000
    assert DB["bookings"]["ABC123"]["status"] == "cancelled"
    assert DB["flights"]["SU2454_0220"]["seats"] == old_seats + 1 # returned seat

def test_get_policy():
    """Verify policy parameters are fetched correctly."""
    rebook_pol = json.loads(get_policy.invoke({"policy_type": "rebooking"}))
    assert rebook_pol["status"] == "success"
    assert rebook_pol["policy"]["fee"] == 40
