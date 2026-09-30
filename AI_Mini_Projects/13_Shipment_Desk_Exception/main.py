"""Command-line runner for the classification and compensation milestone."""

import argparse

from pipeline import process_exception


def main() -> None:
	parser = argparse.ArgumentParser(description="Process a shipment exception")
	parser.add_argument("report", help="Customer's exception report")
	parser.add_argument("shipment_value", type=float, help="Declared shipment value")
	parser.add_argument(
		"--customer-tier",
		choices=("standard", "premium"),
		default="standard",
		help="Customer pricing tier",
	)
	args = parser.parse_args()

	result = process_exception(
		args.report,
		args.shipment_value,
		args.customer_tier,
	)
	print(f"Category: {result['category']}")
	print(f"Compensation: ${result['compensation']:.2f}")
	print(f"Steps: {' -> '.join(result['steps'])}")


if __name__ == "__main__":
	main()
