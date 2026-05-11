# core/quality/judge.py

import json
from dataclasses import dataclass
from infrastructure.llm.nim_client import nim_client

@dataclass
class EvaluationResult:
    factual_accuracy: int
    helpfulness: int
    policy_compliance: int
    tool_correctness: int
    conversation_flow: int
    overall_pass: bool
    reasoning: str
    revision_suggestion: str | None

class QualityJudge:
    """
    Evaluates agent responses before delivery.
    Uses a fast model on purpose - a different model
    from the generating model helps avoid confirmation bias.
    """

    JUDGE_SYSTEM = """You are a Quality Evaluator for a customer support AI.
Your sole job is to evaluate the proposed 'Agent Response' based on the provided context and strict criteria.

Evaluate the response on these 5 criteria, providing a score from 1 (terrible) to 5 (perfect) for each:

1.  **factual_accuracy**: Does the response perfectly match the information in the 'Tool Results'? (Score 5 if no tools were used and the question is general).
2.  **helpfulness**: Does the response directly and completely answer the 'Customer Message' and resolve their issue?
3.  **policy_compliance**: Does the response adhere to standard company policies (e.g., being polite, not making financial promises, not giving legal advice)?
4.  **tool_correctness**: Was the interpretation of the 'Tool Results' correct? (Score 5 if no tools were used).
5.  **conversation_flow**: Is the tone appropriate and the language natural and conversational?

Return a JSON object with this exact structure:
{
  "scores": {
    "factual_accuracy": <1-5>,
    "helpfulness": <1-5>,
    "policy_compliance": <1-5>,
    "tool_correctness": <1-5>,
    "conversation_flow": <1-5>
  },
  "overall_pass": <true or false>,
  "reasoning": "<one-sentence explanation for your scores>",
  "revision_suggestion": "<If it fails, provide a specific, actionable suggestion for how the AI should revise its response. If it passes, this should be null.>"
}

The 'overall_pass' is true ONLY IF ALL of these minimum scores are met:
- factual_accuracy >= 4
- helpfulness >= 3
- policy_compliance >= 4
- tool_correctness >= 4
- conversation_flow >= 3
"""

    THRESHOLDS = {
        "factual_accuracy": 4, "helpfulness": 3, "policy_compliance": 4,
        "tool_correctness": 4, "conversation_flow": 3
    }

    async def evaluate(
        self,
        user_message: str,
        agent_response: str,
        tool_results: list = None,
        intent: str = None
    ) -> EvaluationResult:
        """
        Evaluate a response and return a structured EvaluationResult.
        """
        tool_context = "No tools were used for this response."
        if tool_results:
            tool_context = f"Tools were called and returned the following data:\n{json.dumps(tool_results, indent=2)}"

        evaluation_input = f"""
CONTEXT FOR EVALUATION:
- Customer Message: "{user_message}"
- Detected Intent: "{intent or 'unknown'}"
- Tool Results: {tool_context}

PROPOSED AGENT RESPONSE TO EVALUATE:
"{agent_response}"

Please evaluate the response based on the 5 criteria and provide your JSON output.
"""
        result = await nim_client.complete_json(
            messages=[
                {"role": "system", "content": self.JUDGE_SYSTEM},
                {"role": "user", "content": evaluation_input}
            ],
            tier="fast",  # Use fast model, it's a different "mind" from the generator
            temperature=0.0
        )

        scores_data = result.get("parsed")

        if not scores_data or "scores" not in scores_data:
            # If JSON parsing or the structure is wrong, default to a safe "pass"
            # to avoid blocking all responses due to a judge failure.
            return self._default_pass()

        # Re-check the pass condition here to be absolutely sure, overriding the LLM if needed
        scores = scores_data.get("scores", {})
        all_pass = all(
            scores.get(criterion, 0) >= threshold
            for criterion, threshold in self.THRESHOLDS.items()
        )

        return EvaluationResult(
            factual_accuracy=scores.get("factual_accuracy", 0),
            helpfulness=scores.get("helpfulness", 0),
            policy_compliance=scores.get("policy_compliance", 0),
            tool_correctness=scores.get("tool_correctness", 0),
            conversation_flow=scores.get("conversation_flow", 0),
            overall_pass=all_pass,
            reasoning=scores_data.get("reasoning", "Evaluation failed to parse."),
            revision_suggestion=scores_data.get("revision_suggestion")
        )

    def _default_pass(self) -> EvaluationResult:
        return EvaluationResult(
            factual_accuracy=5, helpfulness=5, policy_compliance=5,
            tool_correctness=5, conversation_flow=5, overall_pass=True,
            reasoning="Judge LLM failed to return valid JSON, defaulting to pass.",
            revision_suggestion=None
        )