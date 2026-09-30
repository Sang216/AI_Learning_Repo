"""Gradio interface for processing shipment exceptions."""

import gradio as gr

from pipeline import process_exception
from session import generate_daily_summary, get_daily_log, record


def _log_rows() -> list[list[str]]:
	return [
		[
			entry["category"].title(),
			f"${entry['shipment_value']:,.2f}",
			f"${entry['compensation']:,.2f}",
			"Yes" if entry["escalated"] else "No",
		]
		for entry in get_daily_log()
	]


def _format_summary() -> str:
	summary = generate_daily_summary()
	if isinstance(summary, str):
		return summary

	return (
		f"**Exceptions processed:** {summary['total_processed']}  \n"
		f"**Total compensation:** ${summary['total_compensation']:,.2f}  \n"
		f"**Escalations:** {summary['escalated_count']} "
		f"({summary['escalation_rate']:.0%})  \n"
		f"**Costliest category:** {summary['costliest_category'].title()} "
		f"(${summary['costliest_category_total']:,.2f})"
	)


def submit_exception(
	report: str, shipment_value: float, customer_tier: str
) -> tuple[str, str, list[list[str]]]:
	try:
		result = process_exception(report, shipment_value, customer_tier)
	except (TypeError, ValueError) as error:
		return f"## Unable to process exception\n\n{error}", _format_summary(), _log_rows()

	record(result)
	route = "Manager review" if result["escalated"] else "Customer resolution"
	outcome = (
		"## Exception Outcome\n\n"
		f"**Category:** {result['category'].title()}  \n"
		f"**Compensation:** ${result['compensation']:,.2f}  \n"
		f"**Route:** {route}  \n"
		f"**Reason:** {result['escalation_reason']}\n\n"
		f"### {route}\n\n{result['draft']}\n\n"
		"### Processing Steps\n\n"
		+ " -> ".join(result["steps"])
	)
	return outcome, _format_summary(), _log_rows()


with gr.Blocks(title="Shipment Exception Desk") as demo:
	gr.Markdown("# Shipment Exception Desk\nProcess and route shipping exceptions.")

	with gr.Row():
		with gr.Column(scale=1):
			report_input = gr.Textbox(
				label="Exception report",
				placeholder="Describe what happened to the shipment...",
				lines=7,
			)
			shipment_value_input = gr.Number(
				label="Shipment value ($)", minimum=0, value=100
			)
			customer_tier_input = gr.Radio(
				["standard", "premium"], label="Customer tier", value="standard"
			)
			submit_button = gr.Button("Process exception", variant="primary")
		with gr.Column(scale=1):
			outcome_output = gr.Markdown("## Exception Outcome\n\nSubmit a report to begin.")

	gr.Markdown("## Daily Triage Log")
	log_output = gr.Dataframe(
		headers=["Category", "Shipment value", "Compensation", "Escalated"],
		value=_log_rows(),
		interactive=False,
	)

	with gr.Row():
		gr.Markdown("## Daily Summary", scale=1)
		summary_button = gr.Button("Refresh", scale=0)
	summary_output = gr.Markdown(_format_summary())

	submit_button.click(
		fn=submit_exception,
		inputs=[report_input, shipment_value_input, customer_tier_input],
		outputs=[outcome_output, summary_output, log_output],
	)
	summary_button.click(fn=_format_summary, outputs=summary_output)


if __name__ == "__main__":
	demo.launch()
