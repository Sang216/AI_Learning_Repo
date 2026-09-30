"""Daily Triage Log and Daily Summary aggregation for the Shipment Exception Desk."""

_daily_log = []


def get_daily_log() -> list[dict]:
	"""Return a copy of today's processed exceptions in submission order."""
	return list(_daily_log)


def record(final_state: dict) -> None:
	"""Append a processed exception's final state to the Daily Triage Log."""
	_daily_log.append(final_state)

def generate_daily_summary() -> str:
	"""Aggregate the Daily Triage Log: totals, escalation rate, and costliest category.

	Costliest category is ranked by *total* compensation paid out, so a
	category with many small payouts can outrank one with a single large one.
	"""
	if not _daily_log:
		return "No exceptions processed today."

	total_processed = len(_daily_log)
	total_compensation = sum(record["compensation"] for record in _daily_log)
	escalated_count = sum(1 for record in _daily_log if record["escalated"])
	escalation_rate = escalated_count / total_processed

	category_totals: dict[str, float] = {}
	for record in _daily_log:
		category_totals[record["category"]] = (
			category_totals.get(record["category"], 0.0) + record["compensation"]
		)

	costliest_category, costliest_category_total = max(
		category_totals.items(), key=lambda item: item[1]
	)

	return {
		"total_processed": total_processed,
		"total_compensation": total_compensation,
		"escalated_count": escalated_count,
		"escalation_rate": escalation_rate,
		"category_totals": category_totals,
		"costliest_category": costliest_category,
		"costliest_category_total": costliest_category_total,
	}
