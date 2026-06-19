# apps/web/app.py
"""
Nexus Bank — Hybrid Support Portal
Combines traditional button-based actions with an intelligent LLM orchestrator.
"""

import streamlit as st
import requests
import json
import uuid
import sqlite3
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "nexus_fintech.db")

st.set_page_config(
    page_title="Nexus Bank | Support Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: #0B0E14;
        color: #E2E8F0;
    }

    /* Modern Headers */
    .bank-header {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(14, 165, 233, 0.05) 100%);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .bank-header h1 {
        background: linear-gradient(135deg, #3B82F6, #0EA5E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .bank-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }

    /* Cards */
    .credit-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card-title { color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .card-number { font-size: 1.2rem; font-weight: 600; letter-spacing: 2px; margin: 8px 0; color: #F8FAFC; }
    
    .status-active { color: #10B981; font-size: 0.8rem; font-weight: 600; }
    .status-frozen { color: #3B82F6; font-size: 0.8rem; font-weight: 600; }
    .status-stolen { color: #EF4444; font-size: 0.8rem; font-weight: 600; }

    /* Transactions */
    .txn-row {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .txn-info { display: flex; flex-direction: column; }
    .txn-merchant { font-weight: 500; font-size: 0.9rem; color: #F8FAFC; }
    .txn-meta { color: #64748B; font-size: 0.75rem; }
    .txn-amount { font-weight: 600; font-size: 1rem; color: #F8FAFC; }
    .txn-status { font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.1); margin-top: 4px; display: inline-block;}

    /* Meta Cards (Chat details) */
    .meta-card {
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    .meta-card h4 { color: #3B82F6; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin: 0 0 8px 0; }
    
    /* Quality Badge */
    .quality-pass { background: rgba(16, 185, 129, 0.15); color: #10B981; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
    .quality-fail { background: rgba(239, 68, 68, 0.15); color: #EF4444; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }

    /* Typing Indicator */
    .typing-indicator { display: flex; gap: 4px; padding: 8px 0; }
    .typing-indicator span { width: 8px; height: 8px; background: #3B82F6; border-radius: 50%; animation: bounce 1.4s infinite both; }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { 0%, 80%, 100% { transform: scale(0); opacity: 0.3; } 40% { transform: scale(1); opacity: 1; } }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── DB Helpers ────────────────────────────────────────────

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_user_data(user_id: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM nexus_users WHERE id = ?", (user_id,)).fetchone()
    cards = conn.execute("SELECT * FROM cards WHERE user_id = ?", (user_id,)).fetchall()
    txns = conn.execute("SELECT * FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 6", (user_id,)).fetchall()
    conn.close()
    return user, cards, txns

def fetch_first_user_id():
    """Helper to auto-login the first user in the DB for the demo."""
    conn = get_db_connection()
    user = conn.execute("SELECT id FROM nexus_users LIMIT 1").fetchone()
    conn.close()
    return user["id"] if user else str(uuid.uuid4())


# ── Session State Init ────────────────────────────────────

if "user_id" not in st.session_state:
    st.session_state.user_id = fetch_first_user_id()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Nexus Bank Support. How can I help you today?"}]
if "action_trigger" not in st.session_state:
    st.session_state.action_trigger = None

user_data, cards_data, txns_data = fetch_user_data(st.session_state.user_id)


# ── Main Layout (Hybrid UI) ───────────────────────────────

col1, col2 = st.columns([1, 1.8], gap="large")

# ==========================================
# LEFT COLUMN: CLIENT PORTAL (BUTTONS/DATA)
# ==========================================
with col1:
    st.markdown("### 💼 My Accounts")
    
    if user_data:
        st.markdown(f"**Welcome back, {user_data['name']}**  •  Tier: `{user_data['tier']}`")
    
    # Render Cards
    st.markdown("<br><b>Active Cards</b>", unsafe_allow_html=True)
    for card in cards_data:
        status_class = "status-active" if card["status"] == "active" else "status-frozen" if card["status"] == "frozen" else "status-stolen"
        st.markdown(f"""
        <div class="credit-card">
            <div class="card-title">Nexus {card["card_type"].title()}</div>
            <div class="card-number">•••• •••• •••• {card["last_4_digits"]}</div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="{status_class}">{card["status"].upper()}</span>
                <span style="color:#94A3B8; font-size:0.8rem;">Limit: ${card["daily_limit"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render Transactions
    st.markdown("<br><b>Recent Transactions</b>", unsafe_allow_html=True)
    for txn in txns_data:
        date_str = datetime.fromisoformat(txn["timestamp"]).strftime("%b %d") if txn["timestamp"] else "Unknown"
        
        st.markdown(f"""
        <div class="txn-row">
            <div class="txn-info">
                <span class="txn-merchant">{txn["merchant_name"]}</span>
                <span class="txn-meta">{date_str} • {txn["category"].title()}</span>
                <span class="txn-status">{txn["status"].upper()}</span>
            </div>
            <div class="txn-amount">${txn["amount"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons right below the transaction
        cols = st.columns([1, 1, 1])
        if txn["status"] == "cleared":
            if cols[0].button("Dispute", key=f"disp_{txn['id']}", help="Dispute this charge"):
                st.session_state.action_trigger = {
                    "intent": "submit_dispute",
                    "txn_id": txn["id"],
                    "msg": f"I want to dispute the charge at {txn['merchant_name']} for ${txn['amount']}."
                }
                st.rerun()
                
            if txn["category"] == "fee" and cols[1].button("Waive Fee", key=f"waive_{txn['id']}", help="Request fee waiver"):
                st.session_state.action_trigger = {
                    "intent": "fee_waiver",
                    "txn_id": txn["id"],
                    "msg": f"Please waive the {txn['merchant_name']} fee of ${txn['amount']}."
                }
                st.rerun()


# ==========================================
# RIGHT COLUMN: INTELLIGENT CHAT
# ==========================================
with col2:
    st.markdown("""
    <div class="bank-header">
        <h1>Nexus Support</h1>
        <p>AI Orchestrator with Button-Bypass & Episodic Memory</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat History Container
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🏦"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    if meta.get("escalated"):
                        passed_badge = '<span class="quality-fail" style="background: rgba(220, 38, 38, 0.15); color: #DC2626;">🚨 Escalated</span>'
                    elif meta.get("quality_passed") is True:
                        passed_badge = '<span class="quality-pass">✓ Passed</span>'
                    elif meta.get("quality_passed") is False:
                        passed_badge = '<span class="quality-fail">✗ Failed</span>'
                    else:
                        passed_badge = '<span class="quality-pass" style="background: rgba(148, 163, 184, 0.15); color: #94A3B8;">⏸ Paused</span>'
                    tools_list = meta.get("tools_called", [])
                    tools_str_list = [t if isinstance(t, str) else t.get("tool_name", str(t)) for t in tools_list]
                    tools = ", ".join(tools_str_list) or "None"
                    st.markdown(f"""
                    <div class="meta-card">
                        <h4>Graph Execution</h4>
                        <div><b>Agent:</b> {meta.get("agent_used")} &nbsp;|&nbsp; <b>Tools:</b> {tools} &nbsp;|&nbsp; {passed_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Bottom Actions
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 2])
    if c1.button("🧠 Test: Policy RAG"):
        st.session_state.action_trigger = {"intent": None, "txn_id": None, "msg": "What is the fee waiver policy for Premium members?"}
        st.rerun()
    if c2.button("🚨 Test: Escalation"):
        st.session_state.action_trigger = {"intent": None, "txn_id": None, "msg": "I am going to sue this bank! I demand a lawyer!"}
        st.rerun()
    if c3.button("💾 End Conversation (Save Memory)"):
        with st.spinner("Saving Episodic Memory..."):
            requests.post(
                f"{API_BASE}/end-conversation",
                json={
                    "user_id": st.session_state.user_id,
                    "conversation_history": st.session_state.messages
                }
            )
        st.session_state.messages = [{"role": "assistant", "content": "Memory saved to SupportTickets table. Conversation reset. How can I help?"}]
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    
    # ── Communication logic ────────────────────────────────
    
    def send_chat(user_msg: str, intent_override: str = None, txn_id: str = None):
        st.session_state.messages.append({"role": "user", "content": user_msg})
        
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_msg)
            
            with st.chat_message("assistant", avatar="🏦"):
                status_placeholder = st.empty()
                display_placeholder = st.empty()
                status_placeholder.markdown("<div class='typing-indicator'><span></span><span></span><span></span></div>", unsafe_allow_html=True)

                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
                
                payload = {
                    "message": user_msg,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "conversation_history": history
                }
                if intent_override:
                    payload["intent_override"] = intent_override
                if txn_id:
                    payload["transaction_id"] = txn_id

                try:
                    response = requests.post(f"{API_BASE}/chat/stream", json=payload, stream=True, timeout=60)
                    
                    full_reply = ""
                    meta = {}
                    
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    ev = data.get("type")
                                    if ev == "status":
                                        status_placeholder.caption(f"🔄 {data.get('content')}")
                                    elif ev == "token":
                                        status_placeholder.empty()
                                        full_reply += data.get("content", "")
                                        display_placeholder.markdown(full_reply)
                                    elif ev == "metadata":
                                        meta = data
                                    elif ev == "error":
                                        st.error(f"Graph Error: {data.get('content')}")
                                except Exception:
                                    pass
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_reply,
                        "metadata": meta
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Backend Error: {e}")

    # ── Input Handling ─────────────────────────────────────
    
    if st.session_state.action_trigger:
        trigger = st.session_state.action_trigger
        st.session_state.action_trigger = None
        send_chat(trigger["msg"], trigger["intent"], trigger["txn_id"])

    if prompt := st.chat_input("Ask about your cards, fees, or bank policy..."):
        send_chat(prompt)
