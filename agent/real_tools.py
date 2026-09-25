"""Real-world tools: they hit the actual internet (no API keys needed).

A service desk constantly asks "is it us or them?": is the website up, does DNS
resolve, is the TLS certificate about to expire, is the SaaS vendor having an outage?
"""

import json
import logging
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

# Public Atlassian Statuspage endpoints (same JSON format for all of them).
VENDOR_STATUS_PAGES = {
    "github": "https://www.githubstatus.com/api/v2/status.json",
    "openai": "https://status.openai.com/api/v2/status.json",
    "cloudflare": "https://www.cloudflarestatus.com/api/v2/status.json",
    "atlassian": "https://status.atlassian.com/api/v2/status.json",
    "zoom": "https://status.zoom.us/api/v2/status.json",
    "discord": "https://discordstatus.com/api/v2/status.json",
}
TIMEOUT_S = 8
log = logging.getLogger("itsm.internet")
USER_AGENT = {"User-Agent": "itsm-agent-demo/1.0"}


@tool
def check_website(url: str) -> dict[str, Any]:
    """Check if a website is reachable: HTTP status code and response time. Use for 'is site X down?'."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    log.info("HTTP GET %s", url)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=TIMEOUT_S) as r:  # noqa: S310
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code  # the server answered, just with an error code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"url": url, "reachable": False, "error": str(getattr(e, "reason", e))}
    return {"url": url, "reachable": True, "status_code": status, "up": status < 500,
            "response_ms": round((time.perf_counter() - start) * 1000)}


@tool
def dns_lookup(hostname: str) -> dict[str, Any]:
    """Resolve a hostname to IP addresses (DNS). Use when users report 'site not found' errors."""
    log.info("DNS lookup %s", hostname)
    try:
        ips = sorted({info[4][0] for info in socket.getaddrinfo(hostname, None)})
        return {"hostname": hostname, "resolves": True, "ip_addresses": ips}
    except socket.gaierror as e:
        return {"hostname": hostname, "resolves": False, "error": str(e)}


@tool
def ssl_certificate_expiry(hostname: str) -> dict[str, Any]:
    """Check a website's TLS/SSL certificate: issuer and days until it expires."""
    log.info("TLS handshake %s:443", hostname)
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=TIMEOUT_S) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as tls:
                cert = tls.getpeercert() or {}
    except (OSError, ssl.SSLError) as e:
        return {"hostname": hostname, "valid": False, "error": str(e)}
    expires = datetime.fromtimestamp(ssl.cert_time_to_seconds(cert["notAfter"]), UTC)
    issuer = dict(x[0] for x in cert.get("issuer", ()))
    return {"hostname": hostname, "valid": True, "issuer": issuer.get("organizationName", "?"),
            "expires": expires.date().isoformat(), "days_left": (expires - datetime.now(UTC)).days}


@tool
def vendor_status(vendor: str) -> dict[str, Any]:
    """Get the live public status of a SaaS vendor (github, openai, cloudflare, atlassian, zoom, discord)."""
    url = VENDOR_STATUS_PAGES.get(vendor.lower().strip())
    if url is None:
        return {"error": f"unknown vendor '{vendor}'. Known: {sorted(VENDOR_STATUS_PAGES)}"}
    log.info("Vendor status %s (%s)", vendor, url)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=TIMEOUT_S) as r:  # noqa: S310
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return {"vendor": vendor, "error": f"status page unreachable: {e}"}
    status = data.get("status", {})
    return {"vendor": vendor, "indicator": status.get("indicator"), "description": status.get("description"),
            "updated_at": data.get("page", {}).get("updated_at")}


BUSINESS_HOURS = (8, 18)  # desk-local time, Monday to Friday


def desk_time(now: datetime | None = None) -> str:
    """Desk-local time and whether it is business hours, decided in code (the model must not guess from UTC).

    Time zone: DESK_TIMEZONE (e.g. Asia/Kuala_Lumpur), else the server's local time zone.
    """
    tz = ZoneInfo(os.environ["DESK_TIMEZONE"]) if os.getenv("DESK_TIMEZONE") else None
    local = (now or datetime.now(UTC)).astimezone(tz)
    start, end = BUSINESS_HOURS
    open_now = local.weekday() < 5 and start <= local.hour < end
    return (f"{local.isoformat(timespec='seconds')} ({local:%A}, desk local time). "
            f"Business hours (Mon-Fri {start:02d}:00-{end:02d}:00): {'yes' if open_now else 'no'}.")


@tool
def current_time() -> str:
    """Current desk-local date and time, and whether it is business hours. Call it before applying any
    time-based rule (e.g. 'no restarts during business hours'), and for timestamps in notes."""
    return desk_time()


REAL_TOOLS = [check_website, dns_lookup, ssl_certificate_expiry, vendor_status]
