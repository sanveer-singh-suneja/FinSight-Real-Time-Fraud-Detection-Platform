"""
FinSight Webhook Alert Service.
Sends fraud alerts to Slack, Discord, Microsoft Teams, and generic webhooks.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

from configs.settings import get_settings

logger = structlog.get_logger(__name__)


def _slack_payload(alert_type: str, message: str, fraud_score: float, transaction_id: str) -> dict:
    color = "#ff0000" if fraud_score >= 0.85 else "#ff9900" if fraud_score >= 0.5 else "#36a64f"
    return {
        "attachments": [
            {
                "color": color,
                "title": f"🚨 FinSight Fraud Alert: {alert_type}",
                "text": message,
                "fields": [
                    {"title": "Transaction ID", "value": transaction_id, "short": True},
                    {"title": "Fraud Score", "value": f"{fraud_score:.4f}", "short": True},
                    {"title": "Time", "value": datetime.now(timezone.utc).isoformat(), "short": True},
                ],
                "footer": "FinSight Fraud Detection Platform",
            }
        ]
    }


def _discord_payload(alert_type: str, message: str, fraud_score: float, transaction_id: str) -> dict:
    color = 0xFF0000 if fraud_score >= 0.85 else 0xFF9900 if fraud_score >= 0.5 else 0x36A64F
    return {
        "embeds": [
            {
                "title": f"🚨 FinSight Fraud Alert: {alert_type}",
                "description": message,
                "color": color,
                "fields": [
                    {"name": "Transaction ID", "value": transaction_id, "inline": True},
                    {"name": "Fraud Score", "value": f"{fraud_score:.4f}", "inline": True},
                    {"name": "Time", "value": datetime.now(timezone.utc).isoformat(), "inline": False},
                ],
                "footer": {"text": "FinSight Fraud Detection Platform"},
            }
        ]
    }


def _teams_payload(alert_type: str, message: str, fraud_score: float, transaction_id: str) -> dict:
    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "FF0000" if fraud_score >= 0.85 else "FF9900",
        "summary": f"FinSight Fraud Alert: {alert_type}",
        "sections": [
            {
                "activityTitle": f"🚨 FinSight Fraud Alert: {alert_type}",
                "activitySubtitle": "Real-Time Fraud Detection Platform",
                "text": message,
                "facts": [
                    {"name": "Transaction ID", "value": transaction_id},
                    {"name": "Fraud Score", "value": f"{fraud_score:.4f}"},
                    {"name": "Time", "value": datetime.now(timezone.utc).isoformat()},
                ],
            }
        ],
    }


def _generic_payload(alert_type: str, message: str, fraud_score: float, transaction_id: str, extra: dict | None = None) -> dict:
    return {
        "source": "finsight",
        "alert_type": alert_type,
        "message": message,
        "fraud_score": fraud_score,
        "transaction_id": transaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }


async def _post_webhook(session: aiohttp.ClientSession, url: str, payload: dict) -> bool:
    """POST payload to webhook URL. Returns True on success."""
    if not url:
        return False
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status in (200, 201, 204):
                return True
            body = await resp.text()
            logger.warning("webhook_bad_status", url=url[:50], status=resp.status, body=body[:200])
            return False
    except asyncio.TimeoutError:
        logger.error("webhook_timeout", url=url[:50])
        return False
    except Exception as exc:
        logger.error("webhook_error", url=url[:50], error=str(exc))
        return False


async def send_fraud_alert(
    alert_type: str,
    message: str,
    fraud_score: float,
    transaction_id: str,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """
    Send alert to all configured webhook channels.

    Returns a dict mapping channel name → delivery success.
    """
    settings = get_settings()
    results: dict[str, bool] = {}

    async with aiohttp.ClientSession() as session:
        tasks = []

        if settings.webhook.slack_webhook_url:
            payload = _slack_payload(alert_type, message, fraud_score, transaction_id)
            tasks.append(("slack", _post_webhook(session, settings.webhook.slack_webhook_url, payload)))

        if settings.webhook.discord_webhook_url:
            payload = _discord_payload(alert_type, message, fraud_score, transaction_id)
            tasks.append(("discord", _post_webhook(session, settings.webhook.discord_webhook_url, payload)))

        if settings.webhook.teams_webhook_url:
            payload = _teams_payload(alert_type, message, fraud_score, transaction_id)
            tasks.append(("teams", _post_webhook(session, settings.webhook.teams_webhook_url, payload)))

        if settings.webhook.webhook_url:
            payload = _generic_payload(alert_type, message, fraud_score, transaction_id, extra_data)
            tasks.append(("generic", _post_webhook(session, settings.webhook.webhook_url, payload)))

        if tasks:
            channel_names, coroutines = zip(*tasks)
            deliveries = await asyncio.gather(*coroutines, return_exceptions=False)
            results = dict(zip(channel_names, deliveries))

    if not results:
        logger.info("no_webhook_channels_configured")
    else:
        logger.info("alerts_sent", results=results)

    return results


async def send_system_alert(
    alert_type: str,
    message: str,
    severity: str = "HIGH",
) -> dict[str, bool]:
    """Send a system-level (non-transaction) alert."""
    return await send_fraud_alert(
        alert_type=alert_type,
        message=message,
        fraud_score=1.0 if severity == "CRITICAL" else 0.8,
        transaction_id="system",
        extra_data={"severity": severity},
    )
