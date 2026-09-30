"""Classification and compensation pipeline for shipment exceptions."""

from chains import classify_chain, draft_email_chain, escalate_chain
from tools import (
	calculate_damage_compensation,
	calculate_delay_compensation,
	calculate_lost_compensation,
)

COMPENSATION_CALCULATORS = {
	"delayed": calculate_delay_compensation,
	"damaged": calculate_damage_compensation,
	"lost": calculate_lost_compensation,
}

STANDARD_ESCALATION_THRESHOLD = 200.0
PREMIUM_ESCALATION_THRESHOLD = 100.0


def _classify_report(report: str) -> str:
	result = classify_chain.invoke({"report": report})
	category = result.strip().lower().strip(".\n ")
	return category if category in COMPENSATION_CALCULATORS else "unknown"


def _draft_message(chain, inputs):
	return chain.invoke(inputs).strip()


def process_exception(
	report: str,
	shipment_value: float,
	customer_tier: str = "standard",
) -> dict:
	"""Classify, compensate, route, and draft a response for an exception."""
	if not isinstance(report, str) or not report.strip():
		raise ValueError("report must be a non-empty string")
	if customer_tier not in {"standard", "premium"}:
		raise ValueError("customer_tier must be 'standard' or 'premium'")

	category = _classify_report(report)
	calculator = COMPENSATION_CALCULATORS.get(category)
	compensation = calculator(shipment_value) if calculator else {
		"category": "unknown",
		"compensation": 0.0,
		"reason": "The report could not be classified.",
	}
	compensation_amount = compensation["compensation"]

	threshold = (
		PREMIUM_ESCALATION_THRESHOLD
		if customer_tier == "premium"
		else STANDARD_ESCALATION_THRESHOLD
	)
	escalated = category == "unknown" or compensation_amount > threshold
	escalation_reason = (
		"unknown category"
		if category == "unknown"
		else "compensation exceeds threshold" if escalated else "within automatic-resolution threshold"
	)

	chain_inputs = {
		"report": report,
		"category": category,
		"shipment_value": float(shipment_value),
		"compensation": compensation_amount,
		"customer_tier": customer_tier,
	}
	drafting_chain = escalate_chain if escalated else draft_email_chain
	draft = _draft_message(drafting_chain, chain_inputs)

	return {
		"category": category,
		"shipment_value": float(shipment_value),
		"customer_tier": customer_tier,
		"compensation": compensation_amount,
		"compensation_details": compensation,
		"escalated": escalated,
		"escalation_reason": escalation_reason,
		"threshold": threshold,
		"draft": draft,
		"steps": [
			"classified",
			"compensation calculated",
			"escalation decision made",
			"manager note drafted" if escalated else "customer email drafted",
		],
	}
