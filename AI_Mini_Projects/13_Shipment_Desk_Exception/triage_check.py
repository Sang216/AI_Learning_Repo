"""Scripted check that exercises all four category/escalation branches end-to-end."""

from pipeline import process_exception
from session import generate_daily_summary, record

SCENARIOS = [
	{
		"name": "mild delay",
		"report": "My package arrived two days late, but everything inside was fine.",
		"shipment_value": 100.0,
		"customer_tier": "standard",
		"expected_category": "delayed",
		"expected_escalated": False,
	},
	{
		"name": "high-value loss",
		"report": "The carrier lost my shipment entirely -- it never arrived and tracking has gone dead.",
		"shipment_value": 2000.0,
		"customer_tier": "standard",
		"expected_category": "lost",
		"expected_escalated": True,
	},
	{
		"name": "minor damage claim",
		"report": "One item in the box arrived with a small dent, but it's still fully usable.",
		"shipment_value": 100.0,
		"customer_tier": "standard",
		"expected_category": "damaged",
		"expected_escalated": False,
	},
	{
		"name": "garbled unclassifiable report",
		"report": "asdkjh qwer??? !!zzz 12345 ####",
		"shipment_value": 10.0,
		"customer_tier": "standard",
		"expected_category": "unknown",
		"expected_escalated": True,
	},
]


def main() -> None:
	all_passed = True
	for scenario in SCENARIOS:
		result = process_exception(
			scenario["report"],
			scenario["shipment_value"],
			scenario["customer_tier"],
		)
		record(result)

		category_ok = result["category"] == scenario["expected_category"]
		escalation_ok = result["escalated"] == scenario["expected_escalated"]
		passed = category_ok and escalation_ok
		all_passed = all_passed and passed

		status = "PASS" if passed else "FAIL"
		print(f"[{status}] {scenario['name']}")
		print(f"  category:     {result['category']} (expected {scenario['expected_category']})")
		print(f"  escalated:    {result['escalated']} (expected {scenario['expected_escalated']})")
		print(f"  compensation: ${result['compensation']:.2f}")
		print()

	print("=" * 40)
	print("ALL SCENARIOS PASSED" if all_passed else "SOME SCENARIOS FAILED")
	print("=" * 40)

	print("\nDaily Summary:")
	print(generate_daily_summary())


if __name__ == "__main__":
	main()
