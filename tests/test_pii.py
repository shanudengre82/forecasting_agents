"""Unit tests for PII & secret detection/redaction, including zero false positives on time-series facts."""

import json

import pytest

from fcagents.guardrails.pii import redact_pii


@pytest.mark.parametrize(
    "raw, expected_placeholder, entity_type",
    [
        ("Contact alice.smith@example.com for details.", "[EMAIL]", "EMAIL"),
        ("Call +1-555-234-5678 or (555) 987-6543.", "[PHONE]", "PHONE"),
        ("SSN is 123-45-6789.", "[SSN]", "SSN"),
        ("Card 4532-0151-1283-0366 charged.", "[CREDIT_CARD]", "CREDIT_CARD"),
        ("Server at 192.168.1.42 failed.", "[IP_ADDRESS]", "IP_ADDRESS"),
        ("IPv6 host 2001:db8::8a2e:370:7334 responded.", "[IP_ADDRESS]", "IP_ADDRESS"),
        ("IBAN DE89370400440532013000 verified.", "[IBAN]", "IBAN"),
        ("Token hf_abcdefghijklmnopqrstuvwxyz1234 leaked.", "[SECRET]", "SECRET"),
        ("Key sk-proj-abcdefghijklmnopqrstuvwxyz1234 leaked.", "[SECRET]", "SECRET"),
        ("Auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "[SECRET]", "SECRET"),
    ],
)
def test_redacts_pii_and_secrets(raw, expected_placeholder, entity_type):
    res = redact_pii(raw)
    assert res.has_pii
    assert expected_placeholder in res.redacted_text
    assert res.counts[entity_type] >= 1


def test_multiple_entities_redacted_in_one_string():
    text = "Email bob@corp.io, phone 415-555-0199, key hf_1234567890abcdef1234."
    res = redact_pii(text)
    assert "bob@corp.io" not in res.redacted_text
    assert "415-555-0199" not in res.redacted_text
    assert "hf_1234567890abcdef1234" not in res.redacted_text
    assert res.counts == {"EMAIL": 1, "PHONE": 1, "SECRET": 1}


def test_luhn_rejects_non_credit_card_numbers():
    # 16-digit number that fails Luhn checksum
    text = "order_id=1234567890123456 count=9999999999999999"
    res = redact_pii(text)
    assert not res.has_pii
    assert res.redacted_text == text


def test_zero_false_positives_on_time_series_facts():
    facts = {
        "n_points": 8760,
        "start": "2024-01-01T00:00:00",
        "end": "2024-12-31T23:00:00",
        "frequency": "h (hourly)",
        "mean": 1234.5678,
        "std": 456.7891,
        "min": 0.0,
        "min_at": "2024-02-14T03:00:00",
        "max": 987654321.1234,
        "max_at": "2024-07-04T18:30:45",
        "candidate_seasonal_periods": [24, 168],
        "series_id": "store_1001",
    }
    payload = json.dumps(facts)
    res = redact_pii(payload)
    assert not res.has_pii
    assert res.redacted_text == payload
