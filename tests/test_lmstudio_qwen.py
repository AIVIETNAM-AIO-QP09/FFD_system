from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

system_prompt = """
You are a Fraud Analyst Co-pilot.

Your task is to convert provided structured evidence and retrieved similar cases into a concise analyst-facing investigation brief.

Strict rules:
- Use only the provided evidence.
- Do not use external fraud-domain assumptions.
- Do not say a channel, device, location, or amount is risky unless the provided evidence explicitly says so.
- Do not invent facts.
- Do not expose chain-of-thought.
- Do not make the final fraud decision.
- Do not recommend automatic blocking, fund freezing, or account suspension.
- If evidence is weak, explicitly say evidence is weak.
- Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
- The underlying ML model is a weak ranking signal.
- Output must follow the required format exactly.
"""

user_prompt = """
/no_think

Generate a Fraud Investigation Brief for:

Transaction ID: T4475404
Model Risk Score: 0.0765
Risk Band: High
Recommended Action: Escalate to senior analyst

Evidence:
- Ranked in the top 0.10% by the Cascade model.
- Amount: $9.19 via wire_transfer.
- No strong novelty or behavioral anomaly was extracted from available structured features.

Similar Historical Cases:
1. Dist: 3.34 | Label: Fraud | Amount: $13.92 | Channel: wire_transfer | Device: pos
2. Dist: 3.47 | Label: Legitimate | Amount: $44.73 | Channel: wire_transfer | Device: atm
3. Dist: 3.83 | Label: Legitimate | Amount: $85.69 | Channel: wire_transfer | Device: web

Required format:

Fraud Investigation Brief

Transaction ID:
Model Risk Score:
Risk Band:
Recommended Action:

Key Risk Factors:
1.
2.
3.

Similar Historical Cases:
1.
2.
3.

Why This Was Flagged:

Caveats:
1.
2.
3.

Suggested Analyst Next Step:
"""

response = client.chat.completions.create(
    model="qwen/qwen3-8b:2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.1,
    max_tokens=700,
)

print(response.choices[0].message.content)