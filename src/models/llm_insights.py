"""
src/models/llm_insights.py
--------------------------
LLM-augmented business insights for customer segments and high-risk churn alerts.
Uses OpenAI API when available; falls back to template-based demo mode.
"""

import os
import json
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

APP_MODE = os.getenv("APP_MODE", "demo")
API_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


# ── Prompt Templates ───────────────────────────────────────────────────────────
SEGMENT_PROMPT = """
You are a senior customer success strategist at a telecom company.
Analyze the following customer segment and provide 3 specific, actionable business recommendations.

Segment: {segment_name}
Customer Count: {count:,}
Churn Rate: {churn_rate:.1%}
Avg Monthly Charges: ${avg_monthly:.2f}
Avg Tenure: {avg_tenure:.1f} months
Avg Support Calls: {avg_support:.1f}
Avg Products: {avg_products:.1f}

Respond in JSON with this exact structure:
{{
  "summary": "2-sentence segment summary",
  "risk_level": "High/Medium/Low",
  "recommendations": [
    {{"action": "Action title", "detail": "Specific steps to take", "expected_impact": "Expected outcome"}},
    {{"action": "Action title", "detail": "Specific steps to take", "expected_impact": "Expected outcome"}},
    {{"action": "Action title", "detail": "Specific steps to take", "expected_impact": "Expected outcome"}}
  ]
}}
"""

HIGH_RISK_PROMPT = """
You are a retention specialist. A customer has a high churn probability.
Generate a personalized retention strategy.

Customer Profile:
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges:.2f}
- Contract: {contract}
- Support Calls (last period): {support_calls}
- Number of Products: {num_products}
- Churn Probability: {churn_prob:.0%}

Respond in JSON:
{{
  "risk_summary": "Why this customer is at risk (2 sentences)",
  "retention_offer": "Specific offer to make",
  "contact_script": "What the retention agent should say (3-4 sentences)",
  "urgency": "Immediate/This week/This month"
}}
"""


# ── Demo Mode Responses ────────────────────────────────────────────────────────
DEMO_SEGMENT_RESPONSES = {
    "Low-Risk Loyalists": {
        "summary": "These long-tenured customers show strong loyalty with low churn rates. They represent your most stable revenue base.",
        "risk_level": "Low",
        "recommendations": [
            {"action": "Launch Loyalty Rewards Program",
             "detail": "Introduce tiered rewards for customers with 24+ months tenure — exclusive discounts, priority support, and early product access.",
             "expected_impact": "Increase NPS by 15% and reduce churn by 2–3% among this segment."},
            {"action": "Upsell Premium Services",
             "detail": "These customers trust the brand — offer bundled upgrades (e.g., Fiber + Streaming + Security package) with a loyalty discount.",
             "expected_impact": "Expected 12–18% uptake rate; increases ARPU by ~$15/month."},
            {"action": "Create Referral Incentives",
             "detail": "Satisfied loyal customers are your best advocates. Deploy a refer-a-friend program with bill credits.",
             "expected_impact": "Estimated 10–15% of customers refer at least one new customer."},
        ]
    },
    "High-Value Engaged": {
        "summary": "High-spending, multi-product customers with moderate tenure. They generate significant revenue but require continued engagement.",
        "risk_level": "Medium",
        "recommendations": [
            {"action": "Assign Dedicated Account Managers",
             "detail": "High-value customers benefit from white-glove service. Assign dedicated reps for proactive quarterly check-ins.",
             "expected_impact": "Reduce churn by 5–8% in this segment."},
            {"action": "Proactive Usage Insights",
             "detail": "Send monthly personalized usage reports showing value delivered vs. cost paid.",
             "expected_impact": "Reinforces perceived value; shown to reduce cancellation intent by 20%."},
            {"action": "Exclusive Beta Access",
             "detail": "Invite this segment to beta-test new features, making them feel valued and increasing switching costs.",
             "expected_impact": "Higher engagement scores and 3–5% churn reduction."},
        ]
    },
    "At-Risk Churners": {
        "summary": "Short-tenure, high-support-call customers on month-to-month contracts with above-average charges. Highest churn risk segment.",
        "risk_level": "High",
        "recommendations": [
            {"action": "Immediate Outreach & Retention Offer",
             "detail": "Trigger automated flag for retention team. Offer 20% discount or contract lock-in with added perks within 48 hours of identification.",
             "expected_impact": "Estimated 30–40% of at-risk customers can be saved with timely intervention."},
            {"action": "Resolve Support Issues Urgently",
             "detail": "High support call volume indicates unresolved pain points. Escalate open tickets; assign senior tech agents.",
             "expected_impact": "Each resolved issue reduces churn probability by ~8%."},
            {"action": "Contract Conversion Campaign",
             "detail": "Offer significant monthly savings (15–20%) in exchange for switching to a 1 or 2-year contract.",
             "expected_impact": "Reduces churn risk by 40–60% for customers who convert."},
        ]
    },
    "New Uncertain": {
        "summary": "Recent customers who haven't yet formed strong brand loyalty. Their experience in the first 90 days determines long-term retention.",
        "risk_level": "Medium",
        "recommendations": [
            {"action": "Onboarding Excellence Program",
             "detail": "Deploy structured 30-60-90 day onboarding: welcome call, setup checklist, feature discovery emails, and first-value milestone celebration.",
             "expected_impact": "Customers who complete onboarding churn 45% less in year 1."},
            {"action": "Early Warning Trigger System",
             "detail": "Flag customers with 0 logins in 2 weeks, 2+ support calls in month 1, or no product adoption.",
             "expected_impact": "Catching issues early saves 3x more customers than late intervention."},
            {"action": "First Contract Incentive",
             "detail": "Offer a 'graduate discount' — reward new customers for committing to a 1-year contract after 60 days with a service credit.",
             "expected_impact": "Expected 25–35% conversion to annual contracts."},
        ]
    }
}

DEMO_HIGH_RISK_RESPONSE = {
    "risk_summary": "This customer shows multiple high-risk signals: frequent support contacts suggest unresolved frustration, and their month-to-month contract offers no switching friction. With above-average monthly charges relative to their short tenure, they likely feel they aren't getting adequate value.",
    "retention_offer": "Offer a 25% discount on current plan locked into a 1-year contract, plus one month free premium tech support service.",
    "contact_script": "Hi [Name], I'm reaching out because we truly value you as a customer and want to make sure you're getting the most from your service. I noticed you've had some recent support needs and wanted to personally follow up. I'd love to offer you an exclusive loyalty discount of 25% off your current plan, plus a complimentary tech support upgrade. Can we take a few minutes to make sure everything is running smoothly for you?",
    "urgency": "Immediate"
}


# ── LLM Client ─────────────────────────────────────────────────────────────────
def _call_llm(prompt: str) -> dict:
    """Call OpenAI API or return demo response."""
    if APP_MODE == "demo" or not API_KEY or not OPENAI_AVAILABLE:
        return None  # Caller handles demo fallback

    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    return json.loads(text)


# ── Public API ─────────────────────────────────────────────────────────────────
def get_segment_insights(segment_profile: dict) -> dict:
    """
    Generate business recommendations for a customer segment.

    Args:
        segment_profile: dict with keys — segment_name, count, churn_rate,
                         avg_monthly, avg_tenure, avg_support, avg_products
    Returns:
        dict with summary, risk_level, recommendations
    """
    prompt = SEGMENT_PROMPT.format(**segment_profile)
    result = _call_llm(prompt)

    if result is None:
        name = segment_profile.get("segment_name", "At-Risk Churners")
        return DEMO_SEGMENT_RESPONSES.get(name, DEMO_SEGMENT_RESPONSES["At-Risk Churners"])

    return result


def get_retention_strategy(customer: dict, churn_prob: float) -> dict:
    """
    Generate a personalized retention strategy for a high-risk customer.

    Args:
        customer: dict with customer features
        churn_prob: float probability of churn
    Returns:
        dict with risk_summary, retention_offer, contact_script, urgency
    """
    profile = {
        "tenure":          customer.get("tenure", "N/A"),
        "monthly_charges": customer.get("monthly_charges", 0),
        "contract":        customer.get("contract", "Month-to-month"),
        "support_calls":   customer.get("support_calls", 0),
        "num_products":    customer.get("num_products", 1),
        "churn_prob":      churn_prob,
    }
    prompt = HIGH_RISK_PROMPT.format(**profile)
    result = _call_llm(prompt)
    return result if result else DEMO_HIGH_RISK_RESPONSE


def generate_executive_summary(metrics: dict) -> str:
    """Generate a plain-English executive summary of churn model results."""
    if APP_MODE == "demo" or not API_KEY or not OPENAI_AVAILABLE:
        return (
            f"📊 **Churn Model Executive Summary**\n\n"
            f"Our churn prediction model achieved **{metrics.get('accuracy', 0):.1%} accuracy** "
            f"with an AUC-ROC of **{metrics.get('auc_roc', 0):.3f}**, indicating strong discriminative ability. "
            f"Of {metrics.get('total_customers', 0):,} customers analyzed, approximately "
            f"**{metrics.get('churn_rate', 0):.1%}** are predicted to churn in the next billing cycle. "
            f"Targeting the top 20% highest-risk customers with retention offers could prevent an estimated "
            f"**${metrics.get('revenue_at_risk', 0):,.0f}** in monthly revenue loss. "
            f"Key churn drivers include: short tenure, month-to-month contracts, high support call volume, "
            f"and elevated monthly charges without bundled services."
        )

    prompt = f"""
    Generate a concise 3-paragraph executive summary for a board presentation about customer churn.
    Model Performance: Accuracy={metrics.get('accuracy', 0):.1%}, AUC-ROC={metrics.get('auc_roc', 0):.3f}
    Churn Rate: {metrics.get('churn_rate', 0):.1%}
    Revenue at Risk: ${metrics.get('revenue_at_risk', 0):,.0f}/month
    Top Churn Drivers: {metrics.get('top_features', [])}
    Write in clear business language suitable for non-technical executives.
    """
    if OPENAI_AVAILABLE and API_KEY:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=500
        )
        return response.choices[0].message.content
    return "Executive summary unavailable — please configure OPENAI_API_KEY."
