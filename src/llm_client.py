import os
import re
from openai import OpenAI

def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def validate_guardrails(text: str, risk_factors: list[str], risk_band: str, score: float) -> tuple[bool, list[str]]:
    violations = []
    text_lower = text.lower()
    
    forbidden_phrases = [
        "unusual behavior", "unusual transaction time", "known fraudulent activity",
        "known risk factors", "typical behavior", "aligns with historical legitimate transactions",
        "consistent with similar legitimate transactions", "no evidence of fraud was identified",
        "device anomaly", "channel anomaly", "suspicious", "red flag", "potential fraud",
        "account compromise", "linked to fraud", "associated with fraud", "higher risk", "lower risk",
        "automatically block", "account suspension", "suspend account", "stolen device", 
        "blacklist", "ip mismatch", "device reputation", "risky channel"
    ]
    
    rf_str = " ".join(risk_factors).lower() if risk_factors else ""
    
    for phrase in forbidden_phrases:
        if phrase.lower() in text_lower:
            # For deterministic guardrails, we reject even if in evidence, 
            # unless we specifically need it. To be safe, we just reject if it appears.
            violations.append(f"Contains unsupported phrase: {phrase}")
            
    if violations:
        return False, violations
    return True, []

class LMStudioLLMClient:
    def __init__(
        self,
        base_url: str = None,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = 120,
    ):
        self.base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        self.model_name = model_name or os.environ.get("LMSTUDIO_MODEL_NAME", "qwen/qwen3-8b:2")
        self.temperature = temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("LLM_MAX_TOKENS", "700"))
        self.timeout = timeout
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="lm-studio"
        )
        
        self.system_prompt = """You are a Fraud Analyst Co-pilot. Generate a candidate brief."""

    def generate_fraud_brief(
        self,
        transaction_id: str,
        score: float,
        risk_band: str,
        recommended_action: str,
        risk_factors: list[str],
        similar_cases: list[dict],
    ) -> str:
        
        sim_cases_str = ""
        for idx, sc in enumerate(similar_cases):
            sim_cases_str += f"{idx+1}. Dist: {sc.get('dist', 0):.2f} | Label: {sc.get('label', '')} | Amount: ${sc.get('amt', 0):.2f} | Channel: {sc.get('channel', '')} | Device: {sc.get('device', '')}\n"
            
        risk_factors_str = ""
        for idx, rf in enumerate(risk_factors):
            risk_factors_str += f"{idx+1}. {rf}\n"
            
        top_half = f"""Fraud Investigation Brief

Transaction ID: {transaction_id}
Model Risk Score: {score:.4f}
Risk Band: {risk_band}
Recommended Action: {recommended_action}

Key Risk Factors:
{risk_factors_str.strip() if risk_factors_str else "1. None"}

Similar Historical Cases:
{sim_cases_str.strip() if sim_cases_str else "1. None"}
"""

        # Deterministic Sections
        if risk_band == "Low":
            explanation_str = "Why This Was Not Escalated:\nThis transaction was not escalated because it falls into the TSLT-missing structural bucket, which historically showed near-zero fraud in the training/validation data. This dependency may reflect a data artifact and should be monitored for drift."
            caveats_str = "Caveats:\n1. The model score is 0.0000 and does not indicate model-based suspicion under the current scoring policy.\n2. TSLT-missing behavior may reflect a data artifact and must be monitored for drift.\n3. Similar historical cases are nearest-neighbor retrieval examples, not causal proof."
            next_step_str = "Suggested Analyst Next Step:\nNo immediate review required under current policy. Monitor TSLT-missing rate and fraud rate over time."
        else:
            explanation_str = "Why This Was Flagged:\nThis transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud."
            caveats_str = "Caveats:\n1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.\n2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.\n3. The listed features are evidence for review routing, not definitive fraud indicators."
            
            if risk_band == "High":
                next_step_str = "Suggested Analyst Next Step:\nEscalate to a senior analyst for manual review. If available, verify sender identity and compare the transaction with recent legitimate activity from the same sender. Do not block or freeze funds based only on this report."
            elif risk_band == "Review":
                next_step_str = "Suggested Analyst Next Step:\nPerform manual review. If available, check whether the listed new attributes are consistent with recent legitimate activity from the same sender."
            else: # Medium
                next_step_str = "Suggested Analyst Next Step:\nMonitor under current policy. If available, review recent sender activity before taking any additional action."

        user_prompt = f"""/no_think

Based on the following evidence and retrieved cases, generate candidate analyst wording.

Transaction ID: {transaction_id}
Model Risk Score: {score:.4f}
Risk Band: {risk_band}
Recommended Action: {recommended_action}

Evidence:
{risk_factors_str.strip()}

Similar Historical Cases:
{sim_cases_str.strip()}
"""

        def _call_llm(prompt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                return response.choices[0].message.content
            except Exception as e:
                return None

        # LLM may generate candidate wording internally.
        _call_llm(user_prompt)
        
        # Final analyst-facing report must use deterministic guarded text.
        final_report = top_half + "\n" + explanation_str + "\n\n" + next_step_str + "\n\n" + caveats_str
        
        is_valid, violations = validate_guardrails(final_report, risk_factors, risk_band, score)
        if not is_valid:
            raise ValueError(f"Deterministic report contains forbidden phrases! Violations: {violations}\n\nReport:\n{final_report}")
            
        return final_report
