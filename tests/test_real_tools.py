"""Level 1 - real-world tools, with the network mocked (plus live checks under -m live)."""

import io
import json
import socket
import urllib.error

import pytest

from agent import real_tools as rt


class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_website_up_and_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt.urllib.request, "urlopen", lambda req, timeout: FakeResponse(b""))
    out = rt.check_website.invoke({"url": "example.com"})
    assert out["url"] == "https://example.com" and out["up"] is True

    def fail(req, timeout):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(rt.urllib.request, "urlopen", fail)
    assert rt.check_website.invoke({"url": "https://nope.invalid"})["reachable"] is False


def test_vendor_status(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"status": {"indicator": "minor", "description": "Partial outage"}, "page": {"updated_at": "t"}}
    monkeypatch.setattr(rt.urllib.request, "urlopen", lambda req, timeout: FakeResponse(json.dumps(body).encode()))
    out = rt.vendor_status.invoke({"vendor": "GitHub"})
    assert out["indicator"] == "minor" and out["description"] == "Partial outage"
    assert "unknown vendor" in rt.vendor_status.invoke({"vendor": "myspace"})["error"]


def test_dns_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", ("93.184.216.34", 0))])
    assert rt.dns_lookup.invoke({"hostname": "example.com"})["ip_addresses"] == ["93.184.216.34"]

    def fail(h, p):
        raise socket.gaierror("not found")

    monkeypatch.setattr(rt.socket, "getaddrinfo", fail)
    assert rt.dns_lookup.invoke({"hostname": "nope.invalid"})["resolves"] is False


@pytest.mark.live
def test_live_internet() -> None:
    assert rt.check_website.invoke({"url": "https://www.github.com"})["up"]
    assert rt.dns_lookup.invoke({"hostname": "github.com"})["resolves"]
    assert rt.ssl_certificate_expiry.invoke({"hostname": "github.com"})["days_left"] > 0
    assert rt.vendor_status.invoke({"vendor": "github"})["indicator"] in ("none", "minor", "major", "critical")


@pytest.mark.parametrize(("utc", "expected"), [
    ("2026-09-25T15:21:33+00:00",  # 15:21 UTC looks like office hours; at the desk it is 23:21
     "2026-09-25T23:21:33+08:00 (Friday, desk local time). Business hours (Mon-Fri 08:00-18:00): no."),
    ("2026-09-25T03:00:00+00:00", "Business hours (Mon-Fri 08:00-18:00): yes."),
    ("2026-09-26T03:00:00+00:00", "Business hours (Mon-Fri 08:00-18:00): no."),  # Saturday
])
def test_business_hours_are_decided_in_desk_local_time(utc: str, expected: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    monkeypatch.setenv("DESK_TIMEZONE", "Asia/Kuala_Lumpur")
    assert rt.desk_time(datetime.fromisoformat(utc)).endswith(expected)
