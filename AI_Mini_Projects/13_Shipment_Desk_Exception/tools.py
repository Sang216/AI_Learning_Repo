"""Rule-based compensation calculators for shipment exceptions."""


DELAY_COMPENSATION = 25.0
DAMAGE_COMPENSATION_RATE = 0.5


def _validate_shipment_value(shipment_value: float) -> float:
	"""Return a usable shipment value or raise a clear validation error."""
	if isinstance(shipment_value, bool) or not isinstance(
		shipment_value, (int, float)
	):
		raise TypeError("shipment_value must be a number")
	if shipment_value < 0:
		raise ValueError("shipment_value cannot be negative")
	return float(shipment_value)


def calculate_delay_compensation(shipment_value: float) -> dict:
	"""Calculate the fixed compensation for a delayed shipment."""
	_validate_shipment_value(shipment_value)
	return {
		"category": "delayed",
		"compensation": DELAY_COMPENSATION,
		"reason": "Delay compensation is a fixed $25 credit.",
	}


def calculate_damage_compensation(shipment_value: float) -> dict:
	"""Calculate compensation equal to half of the declared shipment value."""
	value = _validate_shipment_value(shipment_value)
	return {
		"category": "damaged",
		"compensation": value * DAMAGE_COMPENSATION_RATE,
		"reason": "Damage compensation is 50% of the declared shipment value.",
	}


def calculate_lost_compensation(shipment_value: float) -> dict:
	"""Calculate full-value compensation for a lost shipment."""
	value = _validate_shipment_value(shipment_value)
	return {
		"category": "lost",
		"compensation": value,
		"reason": "Lost shipment compensation is 100% of the declared value.",
	}
