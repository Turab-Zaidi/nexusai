# apps/web/pages/1_📊_Admin_Dashboard.py
"""
NexusAI — Admin Dashboard
Real-time operational metrics and conversation replay.
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="NexusAI — Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #151922 50%, #1A1D29 100%);
    }

    .dashboard-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(78, 205, 196, 0.1) 100%);
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 28px;
        backdrop-filter: blur(12px);
    }
    .dashboard-header h1 {
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .dashboard-header p {
        color: #8B8FA3;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }

    .metric-card {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.18);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        color: #8B8FA3;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    .section-title {
        color: #6C63FF;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(108, 99, 255, 0.2);
    }

    .conv-row {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(108, 99, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .conv-row:hover {
        background: rgba(108, 99, 255, 0.08);
        border-color: rgba(108, 99, 255, 0.3);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────

st.markdown("""
<div class="dashboard-header">
    <h1>📊 Admin Dashboard</h1>
    <p>Real-time operational metrics and conversation tracing</p>
</div>
""", unsafe_allow_html=True)


# ── Controls ──────────────────────────────────────────────

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
with col_ctrl1:
    lookback_hours = st.selectbox("Time Window", [1, 6, 12, 24, 48, 168], index=3, format_func=lambda x: f"Last {x}h" if x < 168 else "Last 7 days")
with col_ctrl2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()


# ── Overview Metrics ──────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_overview(hours):
    try:
        r = requests.get(f"{API_BASE}/analytics/overview", params={"hours": hours}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


overview = fetch_overview(lookback_hours)

if overview:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{overview['total_conversations']}</div>
            <div class="metric-label">Total Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        rate = overview.get('resolution_rate', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{rate:.1f}%</div>
            <div class="metric-label">Resolution Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{overview['escalated']}</div>
            <div class="metric-label">Escalations</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        tokens = overview.get('token_usage', {})
        total_tok = tokens.get('prompt_tokens', 0) + tokens.get('completion_tokens', 0)
        display_tok = f"{total_tok:,}" if total_tok < 100000 else f"{total_tok/1000:.1f}K"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{display_tok}</div>
            <div class="metric-label">Total Tokens</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        llm_calls = tokens.get('total_llm_calls', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{llm_calls}</div>
            <div class="metric-label">LLM Calls</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quality Scores ────────────────────────────────────
    avg_scores = overview.get("avg_quality_scores", {})
    if avg_scores:
        st.markdown('<div class="section-title">Average Quality Scores</div>', unsafe_allow_html=True)
        
        score_cols = st.columns(5)
        dims = ["factual_accuracy", "helpfulness", "policy_compliance", "tool_correctness", "conversation_flow"]
        labels = ["Factual Accuracy", "Helpfulness", "Policy Compliance", "Tool Correctness", "Conversation Flow"]
        
        for col, dim, label in zip(score_cols, dims, labels):
            val = avg_scores.get(dim, 0)
            with col:
                color = "#4ECDC4" if val >= 4 else "#FF9F43" if val >= 3 else "#FF6B6B"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background:none; -webkit-text-fill-color:{color};">{val:.1f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("⏳ Could not connect to the analytics API. Make sure the FastAPI server is running.")


# ── Tool Performance ─────────────────────────────────────

st.markdown('<div class="section-title">Tool Performance</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30)
def fetch_tools(hours):
    try:
        r = requests.get(f"{API_BASE}/analytics/tools", params={"hours": hours}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

tools_data = fetch_tools(lookback_hours)
if tools_data and tools_data.get("tools"):
    tool_cols = st.columns(len(tools_data["tools"]))
    for col, tool in zip(tool_cols, tools_data["tools"]):
        with col:
            rate_pct = tool['success_rate'] * 100
            rate_color = "#4ECDC4" if rate_pct >= 95 else "#FF9F43" if rate_pct >= 80 else "#FF6B6B"
            st.markdown(f"""
            <div class="metric-card" style="padding:14px;">
                <div style="color:#FAFAFA; font-weight:600; font-size:0.85rem; margin-bottom:6px;">
                    {tool['tool_name'].replace('_', ' ').title()}
                </div>
                <div style="color:{rate_color}; font-size:1.4rem; font-weight:700;">{rate_pct:.0f}%</div>
                <div class="metric-label">Success ({tool['total_calls']} calls)</div>
                <div style="color:#8B8FA3; font-size:0.7rem; margin-top:4px;">
                    Avg latency: {tool['avg_latency_ms']:.0f}ms
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.caption("No tool execution data available yet.")


# ── Cache Stats ───────────────────────────────────────────

st.markdown('<div class="section-title">Redis Cache</div>', unsafe_allow_html=True)

@st.cache_data(ttl=15)
def fetch_cache():
    try:
        r = requests.get(f"{API_BASE}/analytics/cache", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

cache_data = fetch_cache()
if cache_data and cache_data.get("enabled"):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("Cache Hits", cache_data.get("hits", 0))
    with cc2:
        st.metric("Cache Misses", cache_data.get("misses", 0))
    with cc3:
        hit_rate = cache_data.get("hit_rate", 0) * 100
        st.metric("Hit Rate", f"{hit_rate:.1f}%")
else:
    st.caption("Redis cache is not connected or has no data.")


# ── Recent Conversations ─────────────────────────────────

st.markdown('<div class="section-title">Recent Conversations</div>', unsafe_allow_html=True)

@st.cache_data(ttl=15)
def fetch_conversations(hours, escalated_only):
    try:
        r = requests.get(
            f"{API_BASE}/admin/conversations",
            params={"hours": hours, "escalated_only": escalated_only, "limit": 20},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

show_escalated = st.checkbox("Show escalated only", value=False)
conv_data = fetch_conversations(lookback_hours, show_escalated)

if conv_data and conv_data.get("conversations"):
    for conv in conv_data["conversations"]:
        esc_badge = "🔴" if conv.get("escalated") else "🟢"
        state = conv.get("current_state", "unknown")
        started = conv.get("started_at", "?")[:19]
        
        with st.expander(f"{esc_badge} {conv['id'][:12]}... — {state} — {started}"):
            st.json(conv)

            if st.button(f"🔍 View Full Trace", key=f"trace_{conv['id']}"):
                try:
                    detail = requests.get(
                        f"{API_BASE}/admin/conversations/{conv['id']}",
                        timeout=10
                    )
                    if detail.status_code == 200:
                        detail_data = detail.json()

                        st.markdown("**State Transitions:**")
                        for t in detail_data.get("state_transitions", []):
                            st.code(f"{t['from_state']} → {t['to_state']}  (by {t.get('agent', '?')})", language=None)

                        st.markdown("**Quality Evaluations:**")
                        for q in detail_data.get("quality_evaluations", []):
                            st.json(q)

                        st.markdown("**Tool Executions:**")
                        for te in detail_data.get("tool_executions", []):
                            st.json(te)

                        st.markdown("**LLM Calls:**")
                        for lc in detail_data.get("llm_calls", []):
                            st.json(lc)
                    else:
                        st.error(f"Failed to load trace: {detail.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.caption("No conversations found in this time window.")


# ── Agent Performance ─────────────────────────────────────

st.markdown('<div class="section-title">Agent Performance</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30)
def fetch_agent_perf(hours):
    try:
        r = requests.get(f"{API_BASE}/admin/agents/performance", params={"hours": hours}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

agent_data = fetch_agent_perf(lookback_hours)
if agent_data and agent_data.get("agents"):
    agent_cols = st.columns(len(agent_data["agents"]))
    for col, agent in zip(agent_cols, agent_data["agents"]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:14px;">
                <div style="color:#FAFAFA; font-weight:600; font-size:0.85rem; margin-bottom:6px;">
                    {agent['agent_name'].replace('_', ' ').title()}
                </div>
                <div style="color:#6C63FF; font-size:1.4rem; font-weight:700;">{agent['total_calls']}</div>
                <div class="metric-label">Total Calls</div>
                <div style="color:#8B8FA3; font-size:0.7rem; margin-top:4px;">
                    Avg: {agent['avg_latency_ms']:.0f}ms · {agent['total_tokens']:,} tokens
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.caption("No agent performance data available yet.")
