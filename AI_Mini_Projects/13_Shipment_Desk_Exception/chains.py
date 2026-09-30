"""Three ready-made LangChain chains: classify, escalate (manager note), and draft email.

Same prompt | model | parser pattern used in Class 2.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import llm

_parser = StrOutputParser()

classify_prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You classify shipment exception reports into exactly one category: "
			"delayed, damaged, lost, or unknown. Reply with only the single "
			"category word, lowercase, no punctuation, no explanation.",
		),
		("human", "Report: {report}"),
	]
)
classify_chain = classify_prompt | llm | _parser

escalate_prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You write short, factual internal notes for a manager reviewing an "
			"escalated shipment exception. Mention the category and the "
			"compensation amount, and state why it needs human review.",
		),
		(
			"human",
			"Customer report: {report}\n"
			"Category: {category}\n"
			"Shipment value: ${shipment_value:.2f}\n"
			"Calculated compensation: ${compensation:.2f}\n"
			"Customer tier: {customer_tier}\n\n"
			"Write the manager note.",
		),
	]
)
escalate_chain = escalate_prompt | llm | _parser

draft_email_prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You write brief, empathetic customer-facing emails resolving a "
			"shipment exception automatically. State the category, apologize "
			"appropriately, and state the compensation amount clearly.",
		),
		(
			"human",
			"Customer report: {report}\n"
			"Category: {category}\n"
			"Shipment value: ${shipment_value:.2f}\n"
			"Approved compensation: ${compensation:.2f}\n"
			"Customer tier: {customer_tier}\n\n"
			"Write the customer email.",
		),
	]
)
draft_email_chain = draft_email_prompt | llm | _parser
