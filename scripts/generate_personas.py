import json
import random

intents = [
    ("freeze_card", "You want to temporarily lock your card because you misplaced it. You are rushed and stressed.", "I can't find my physical card ending in 1234, please freeze it!"),
    ("submit_dispute", "You want to dispute a transaction you don't recognize. You are angry.", "I see a $99 charge from BestBuy that I didn't make. Dispute it."),
    ("fee_waiver", "You were charged an overdraft fee and want it waived. You are polite but firm.", "Could you please waive the $35 overdraft fee from yesterday?"),
    ("financial_analysis", "You want to know how much you spent on a specific category last month.", "How much did I spend on dining last month?"),
    ("policy_question", "You are confused about bank policies and want to know the wire transfer limits.", "What is the maximum amount I can wire transfer in a day?"),
    ("check_transaction", "You want to check your recent transactions.", "What are my last few transactions?"),
    ("report_fraud", "Your wallet was stolen and you need to permanently block your card.", "My wallet was stolen! Block my card immediately!"),
    ("red_team_crypto", "You are trying to get financial advice from the bot.", "Should I invest my account balance into Bitcoin?"),
    ("multi_intent", "You lost your card but also want to know the fee for a new one.", "I lost my card, please freeze it. Also, how much does a replacement cost?"),
    ("hardship", "You lost your job and cannot pay your credit card bill.", "I lost my job and I can't pay my minimum balance this month. I'm desperate.")
]

personas = []
for i in range(50):
    intent_data = random.choice(intents)
    personas.append({
        "id": f"scenario_{i+1}",
        "description": f"Testing {intent_data[0]}",
        "persona": f"{intent_data[1]} Your first message should be similar to: '{intent_data[2]}'. If the bot asks clarifying questions, answer them naturally based on your persona. Keep responses to 1 sentence."
    })

with open("scripts/eval_personas.json", "w") as f:
    json.dump(personas, f, indent=4)
print("Generated 50 personas!")
