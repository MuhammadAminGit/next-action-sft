"""The task as the model sees it.

Two system prompts:

- FULL describes every action and what values it takes. The untrained baselines get it,
  as the most detailed prompt tried.
- COMPACT lists only the action names and the output format. The fine-tuned model is
  trained and evaluated with it.

The full prompt is 710 tokens, 82% of a typical example, all masked out of the loss;
training on it would spend most of the compute re-reading text the model is not learning
from. Having learned the actions from labelled examples, the fine-tuned model does not
need the descriptions.

The descriptions come from ABCD's own ontology, which is what the human agents had on
their dashboard as buttons. They say what each action is for and what values it takes,
but not when to use it; that is the policy the model has to learn.
"""

from __future__ import annotations

ACTION_SPECS: dict[str, str] = {
    # account and identity
    "pull-up-account": "open the customer's account. values: [the customer's full name, or the account id if that is what they gave]",
    "verify-identity": (
        "confirm who the customer is. values: [full name, identifier, identifier], where each "
        "identifier is whatever the customer gave (account id, zip code, phone, email, order id), "
        "or n/a"
    ),
    "validate-purchase": "check a purchase belongs to them. values: [username, email, order id], n/a if not given",
    "update-account": (
        "change the account. values: [one of: add service, extend subscription, "
        "remove service, renew subscription]"
    ),
    "membership": "check membership. values: [guest, bronze, silver or gold]",
    "subscription-status": "look up subscription status. values: []",
    "make-password": "generate a new password. values: []",
    "promo-code": "issue a promo code. values: []",
    "ask-the-oracle": "check the system's verdict on the customer's claim. values: []",
    # orders
    "shipping-status": (
        "record the order's shipping status. values: [delivered, in transit, "
        "order received, or out for delivery]"
    ),
    "update-order": (
        "change the order. values: [the change, e.g. change address, change date, change item, "
        "change method, change order, change price, change time, give credit, waive fee, "
        "by mail, in store, drop off center, cancel shipment]"
    ),
    "make-purchase": "buy a product for the customer. values: [brand and item, e.g. guess jeans]",
    "offer-refund": "refund the customer. values: [amount in dollars, digits only]",
    "record-reason": "note the customer's reason. values: [the reason in a few words]",
    "enter-details": "enter a detail into the system. values: [the detail, e.g. an address, phone number or amount]",
    "notify-team": "escalate. values: [manager, website team or purchasing department]",
    "send-link": "send the customer a link. values: []",
    # troubleshooting
    "log-out-in": "ask the customer to log out and back in. values: []",
    "try-again": "ask the customer to try again. values: []",
    "instructions": "give the customer instructions. values: []",
    # knowledge base
    "search-faq": "open the FAQ. values: []",
    "search-policy": "search the policy FAQ. values: []",
    "search-pricing": "search the pricing FAQ. values: []",
    "search-timing": "search the timing FAQ. values: []",
    "search-membership": "search the membership FAQ. values: []",
    "search-boots": "search the boots FAQ. values: []",
    "search-jacket": "search the jacket FAQ. values: []",
    "search-jeans": "search the jeans FAQ. values: []",
    "search-shirt": "search the shirt FAQ. values: []",
    "select-faq": (
        "choose the FAQ answer. values: [article id: policy_N, pricing_N, timing_N, membership_N, "
        "or <item>_how_N / <item>_other_N for boots, jacket, jeans, shirt; N is 1 to 4]"
    ),
}

FULL_SYSTEM_PROMPT = (
    "You are the action step of a customer service agent for an online clothing store. "
    "Given the conversation so far, decide the single next action the agent takes in the "
    "system, and its values.\n\n"
    "Actions:\n"
    + "\n".join(f"- {name}: {spec}" for name, spec in ACTION_SPECS.items())
    + "\n\nReply with only JSON on one line: "
    '{"action": "<action>", "values": ["<value>", ...]}. '
    "Values are lowercase. Use [] when the action takes none."
)

COMPACT_SYSTEM_PROMPT = (
    "You are the action step of a customer service agent for an online clothing store. "
    "Given the conversation so far, reply with the next action and its values as JSON: "
    '{"action": "<action>", "values": [...]}.\n'
    "Actions: " + ", ".join(ACTION_SPECS)
)

PROMPTS = {"full": FULL_SYSTEM_PROMPT, "compact": COMPACT_SYSTEM_PROMPT}


def render_dialogue(turns: list[tuple[str, str]]) -> str:
    """The conversation up to the action being predicted.

    Earlier actions are included, as the system's own message about them, because the
    agent saw them too, and because what has already been done decides what comes next:
    you verify identity after pulling up the account, not before.
    """
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns)


def messages(
    turns: list[tuple[str, str]], answer: str | None = None, prompt: str = "full"
) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": PROMPTS[prompt]},
        {"role": "user", "content": render_dialogue(turns)},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs
