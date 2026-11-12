import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="ShopperPulse | E-Commerce Purchase Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS & Tailwind Setup
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Light Theme Polish */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background adjustments */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Modern Glass Card */
    .modern-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    .modern-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border-color: #CBD5E1;
    }

    /* Stat Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .stat-card.indigo::before { background: linear-gradient(90deg, #4F46E5, #818CF8); }
    .stat-card.emerald::before { background: linear-gradient(90deg, #059669, #34D399); }
    .stat-card.amber::before { background: linear-gradient(90deg, #D97706, #FBBF24); }
    .stat-card.sky::before { background: linear-gradient(90deg, #0284C7, #38BDF8); }

    /* Custom Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    .badge-indigo { background-color: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-emerald { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-amber { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .badge-slate { background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

    /* Insight Banner */
    .insight-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        color: #064E3B;
        font-size: 0.88rem;
    }
    
    .takeaway-pill {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #6366F1;
        font-size: 0.88rem;
        color: #1E293B;
    }

    /* Clean Streamlit elements */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #0F172A;
    }
    
    /* Streamlit sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Radio options in sidebar */
    div[data-testid="stRadio"] > label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Cache Data Loading
@st.cache_data
def load_data():
    file_path = "online_shoppers_intention.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error(f"Dataset '{file_path}' not found!")
        df = pd.DataFrame()
    return df

# Cache Model Loading
@st.cache_resource
def load_model():
    model_path = "XGBoost pipeline final project.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.warning(f"Could not load ML pipeline: {e}")
            return None
    return None

df = load_data()
model = load_model()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="flex items-center space-x-3 mb-6 p-2">
        <div class="h-10 w-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white text-xl font-bold shadow-md shadow-indigo-100">
            🛍️
        </div>
        <div>
            <h1 class="text-lg font-bold text-slate-800 tracking-tight leading-tight">ShopperPulse</h1>
            <p class="text-xs text-slate-500 font-medium">Purchase Behavior </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["🏠 Home & Data Overview", "📊 Purchase Analysis", "🔮 Purchase Predictor"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr class='my-6 border-slate-200'>", unsafe_allow_html=True)
    
    # Dataset quick facts badge
    if not df.empty:
        total_sessions = len(df)
        buyers = df['Revenue'].sum()
        conv_rate = (buyers / total_sessions) * 100
        
        st.markdown(f"""
        <div class="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs space-y-2">
            <div class="font-bold text-slate-700 flex justify-between items-center">
                <span>Dataset Status</span>
                <span class="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
            </div>
            <div class="flex justify-between text-slate-500">
                <span>Total Sessions:</span>
                <span class="font-semibold text-slate-800">{total_sessions:,}</span>
            </div>
            <div class="flex justify-between text-slate-500">
                <span>Buyers:</span>
                <span class="font-semibold text-emerald-600">{buyers:,}</span>
            </div>
            <div class="flex justify-between text-slate-500">
                <span>Conversion Rate:</span>
                <span class="font-semibold text-indigo-600">{conv_rate:.1f}%</span>
            </div>
            <div class="flex justify-between text-slate-500">
                <span>Model:</span>
                <span class="font-semibold text-slate-700">XGBoost (Tuned)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='mt-8 text-center text-xs text-slate-400'</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME & DATA OVERVIEW
# ==========================================
if page == "🏠 Home & Data Overview":
    # Hero Section
    st.markdown("""
    <div class="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-2xl p-8 mb-8 text-white relative overflow-hidden shadow-xl">
        <div class="relative z-10 max-w-3xl">
            <span class="inline-block bg-indigo-500/30 text-indigo-200 border border-indigo-400/30 text-xs px-3 py-1 rounded-full font-semibold mb-3">
                PROJECT OVERVIEW & DATA DICTIONARY
            </span>
            <h1 class="text-3xl font-extrabold tracking-tight mb-2 text-white">Online Shoppers Purchasing Intention</h1>
            <p class="text-indigo-100 text-sm leading-relaxed mb-4">
                Analyze and understand visitor browsing behaviors on e-commerce storefronts to identify patterns that lead to completed purchases. Built with non-technical business clarity in mind.
            </p>
        </div>
        <div class="absolute -right-12 -bottom-12 w-64 h-64 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Executive KPI Metrics
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-card indigo">
                <span class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Total Browsing Sessions</span>
                <div class="text-2xl font-black text-slate-800 my-1">12,330</div>
                <div class="text-xs text-slate-500 flex items-center">
                    <span class="badge badge-indigo mr-1.5">100%</span> Sampled customer visits
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="stat-card emerald">
                <span class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Completed Purchases</span>
                <div class="text-2xl font-black text-emerald-600 my-1">1,908</div>
                <div class="text-xs text-slate-500 flex items-center">
                    <span class="badge badge-emerald mr-1.5">15.5%</span> Overall conversion rate
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="stat-card amber">
                <span class="text-xs font-semibold uppercase text-slate-400 tracking-wider">No Purchase (Drop-Off)</span>
                <div class="text-2xl font-black text-slate-700 my-1">10,422</div>
                <div class="text-xs text-slate-500 flex items-center">
                    <span class="badge badge-amber mr-1.5">84.5%</span> Browsed without buying
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="stat-card sky">
                <span class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Features Analyzed</span>
                <div class="text-2xl font-black text-sky-600 my-1">17 Inputs</div>
                <div class="text-xs text-slate-500 flex items-center">
                    <span class="badge badge-slate mr-1.5">1 Target</span> Revenue (Purchase: Yes/No)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div class='my-6'></div>", unsafe_allow_html=True)

        # Tabbed Layout for Data Exploration & Column Dictionary
        tab_dict, tab_preview, tab_stats = st.tabs([
            "📖 Comprehensive Column Descriptions",
            "🔍 Interactive Data Explorer",
            "📈 Statistical Summary"
        ])
        
        with tab_dict:
            st.markdown("""
            <div class="mb-4">
                <h3 class="text-lg font-bold text-slate-800">Dataset Column Dictionary</h3>
                <p class="text-xs text-slate-500">Each column explained clearly in simple business terms, categorized by its role in user journey analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            categories = {
                "🛒 1. Visitor Browsing Behavior": [
                    {
                        "col": "Administrative",
                        "type": "Integer",
                        "desc": "Number of account-related pages visited (e.g., account settings, login, profile management).",
                        "importance": "Indicates user commitment to managing an active account on the site."
                    },
                    {
                        "col": "Administrative_Duration",
                        "type": "Float (Seconds)",
                        "desc": "Total time spent by the visitor browsing account or admin pages.",
                        "importance": "Higher values show users active in account or payment preference setup."
                    },
                    {
                        "col": "Informational",
                        "type": "Integer",
                        "desc": "Number of informational pages visited (e.g., About Us, Contact Us, Shipping/Returns FAQ, Store Location).",
                        "importance": "Reflects research behavior and brand trust verification."
                    },
                    {
                        "col": "Informational_Duration",
                        "type": "Float (Seconds)",
                        "desc": "Total time spent reading FAQs, policies, and company information.",
                        "importance": "Can indicate cautious buyers looking for guarantees before purchasing."
                    },
                    {
                        "col": "ProductRelated",
                        "type": "Integer",
                        "desc": "Number of product catalog and item detail pages visited during the session.",
                        "importance": "Direct measure of product interest and catalog browsing depth."
                    },
                    {
                        "col": "ProductRelated_Duration",
                        "type": "Float (Seconds)",
                        "desc": "Total time spent reviewing and exploring products on the store.",
                        "importance": "Longer product browsing often signals higher purchase intent."
                    }
                ],
                "📈 2. Website Engagement Metrics": [
                    {
                        "col": "BounceRates",
                        "type": "Float (0.0 to 0.2)",
                        "desc": "Percentage of visitors who enter the site and immediately leave without performing any interaction.",
                        "importance": "High bounce rates indicate irrelevant landing pages or poor site design."
                    },
                    {
                        "col": "ExitRates",
                        "type": "Float (0.0 to 0.2)",
                        "desc": "Percentage of pageviews on specific pages that were the last in the visitor's browsing session.",
                        "importance": "Identifies pages where potential customers drop out before checkout."
                    },
                    {
                        "col": "PageValues",
                        "type": "Float ($ Value)",
                        "desc": "Average value of the pages visited before completing an e-commerce transaction. Calculated by Google Analytics.",
                        "importance": "The single strongest predictor of purchase intent. High Page Value = high conversion probability."
                    }
                ],
                "📅 3. Time & Seasonality Features": [
                    {
                        "col": "SpecialDay",
                        "type": "Float (0.0 to 1.0)",
                        "desc": "Proximity of the session date to a major holiday or shopping event (e.g., Valentine's Day, Mother's Day). 1.0 means exact day of the event.",
                        "importance": "Captures event-driven urgency (e.g., ordering flowers or gifts in time for delivery)."
                    },
                    {
                        "col": "Month",
                        "type": "Categorical (Feb - Dec)",
                        "desc": "Calendar month in which the session occurred (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec).",
                        "importance": "Highlights holiday shopping peaks such as November (Black Friday / Cyber Monday)."
                    },
                    {
                        "col": "Weekend",
                        "type": "Boolean (True/False)",
                        "desc": "Indicates whether the browsing session occurred on a weekend (Saturday/Sunday) or weekday.",
                        "importance": "Reveals whether customer leisure time translates into higher checkout rates."
                    }
                ],
                "💻 4. Technical & Visitor Context": [
                    {
                        "col": "OperatingSystems",
                        "type": "Categorical ID (1 to 8)",
                        "desc": "Operating system identifier used by visitor (e.g., Windows, macOS, Android, iOS).",
                        "importance": "Helps ensure cross-platform compatibility across mobile and desktop devices."
                    },
                    {
                        "col": "Browser",
                        "type": "Categorical ID (1 to 13)",
                        "desc": "Web browser identifier used by the visitor (e.g., Chrome, Safari, Edge, Firefox).",
                        "importance": "Useful for detecting browser-specific conversion drops or checkout friction."
                    },
                    {
                        "col": "Region",
                        "type": "Categorical ID (1 to 9)",
                        "desc": "Geographical location or territory of the visitor.",
                        "importance": "Allows targeted localized offers and regional delivery logistics planning."
                    },
                    {
                        "col": "TrafficType",
                        "type": "Categorical ID (1 to 20)",
                        "desc": "Source channel that brought the visitor to the store (e.g., Direct, Organic Search, Paid Ads, Social Media).",
                        "importance": "Crucial for marketing ROI: identifying which channels deliver actual buyers."
                    },
                    {
                        "col": "VisitorType",
                        "type": "Categorical",
                        "desc": "Visitor loyalty classification: 'Returning_Visitor', 'New_Visitor', or 'Other'.",
                        "importance": "Shows whether retention marketing or new customer acquisition drives more revenue."
                    }
                ],
                "🎯 5. Target Variable (Goal)": [
                    {
                        "col": "Revenue (Purchase)",
                        "type": "Boolean (True / False)",
                        "desc": "The primary outcome: Did this browsing session result in a finalized purchase? (True = Purchased, False = Did Not Buy).",
                        "importance": "The core business KPI and target prediction variable for machine learning."
                    }
                ]
            }
            
            for category_name, col_list in categories.items():
                with st.expander(category_name, expanded=True):
                    for item in col_list:
                        is_target = "Revenue" in item["col"]
                        card_bg = "bg-emerald-50/60 border-emerald-200" if is_target else "bg-white border-slate-200"
                        badge_color = "badge-emerald" if is_target else "badge-indigo"
                        
                        st.markdown(f"""
                        <div class="p-3 mb-2 rounded-xl border {card_bg} text-sm">
                            <div class="flex items-center justify-between mb-1">
                                <span class="font-bold text-slate-800 text-sm">{item['col']}</span>
                                <span class="badge {badge_color}">{item['type']}</span>
                            </div>
                            <p class="text-slate-600 text-xs mb-1.5">{item['desc']}</p>
                            <p class="text-xs text-indigo-700 font-medium">💡 <b>Business Value:</b> {item['importance']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        with tab_preview:
            st.markdown("<p class='text-sm text-slate-600 mb-2'>Browse, search, and filter the raw dataset sessions below:</p>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                v_filter = st.selectbox("Filter by Visitor Type", ["All"] + list(df['VisitorType'].unique()))
            with c2:
                m_filter = st.selectbox("Filter by Month", ["All"] + list(df['Month'].unique()))
            with c3:
                r_filter = st.selectbox("Filter by Purchase Outcome", ["All", "Purchased Only (True)", "No Purchase (False)"])
                
            filtered_df = df.copy()
            if v_filter != "All":
                filtered_df = filtered_df[filtered_df['VisitorType'] == v_filter]
            if m_filter != "All":
                filtered_df = filtered_df[filtered_df['Month'] == m_filter]
            if r_filter == "Purchased Only (True)":
                filtered_df = filtered_df[filtered_df['Revenue'] == True]
            elif r_filter == "No Purchase (False)":
                filtered_df = filtered_df[filtered_df['Revenue'] == False]
                
            st.dataframe(
                filtered_df.head(100),
                use_container_width=True,
                height=380
            )
            st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df):,} matching rows.")
            
        with tab_stats:
            st.markdown("<p class='text-sm text-slate-600 mb-2'>Summary statistics for browsing metrics across all sessions:</p>", unsafe_allow_html=True)
            numeric_cols = [
                'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
            ]
            summary_stats = df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
            summary_stats.columns = ['Average (Mean)', 'Std Deviation', 'Minimum', 'Median (50%)', 'Maximum']
            st.dataframe(summary_stats.round(2), use_container_width=True)


# ==========================================
# PAGE 2: PURCHASE ANALYSIS (NON-TECHNICAL)
# ==========================================
elif page == "📊 Purchase Analysis":
    st.markdown("""
    <div class="mb-6">
        <span class="badge badge-emerald mb-2">BUSINESS INTELLIGENCE & CONVERSION INSIGHTS</span>
        <h1 class="text-2xl md:text-3xl font-extrabold text-slate-800 tracking-tight">Understanding What Drives Purchases</h1>
        <p class="text-slate-500 text-sm mt-1">
            Intuitive, non-technical visual breakdown of customer sessions. Discover the exact drivers and friction points that separate paying customers from drop-offs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual Row 1: Conversion Rate & Visitor Loyalty
    r1_col1, r1_col2 = st.columns([1, 1])
    
    with r1_col1:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">1. Overall Store Conversion Rate</h4>
            <p class="text-xs text-slate-400 mb-3">What proportion of browsing sessions result in an actual sale?</p>
        </div>
        """, unsafe_allow_html=True)
        
        rev_counts = df['Revenue'].value_counts()
        labels = ['Did Not Purchase', 'Purchased']
        values = [rev_counts[False], rev_counts[True]]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.62,
            marker=dict(colors=['#CBD5E1', '#10B981']),
            textinfo='percent+label',
            textposition='outside',
            insidetextorientation='radial',
            hoverinfo='label+value+percent'
        )])
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> Roughly <b>1 out of every 6.5 visitors (15.5%)</b> makes a purchase. This leaves an 84.5% drop-off opportunity for on-site conversion rate optimization.
        </div>
        """, unsafe_allow_html=True)
        
    with r1_col2:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">2. Who Converts Better: New vs. Returning Visitors?</h4>
            <p class="text-xs text-slate-400 mb-3">Comparing purchase probability across visitor types.</p>
        </div>
        """, unsafe_allow_html=True)
        
        visitor_stats = df.groupby('VisitorType')['Revenue'].agg(['count', 'mean']).reset_index()
        visitor_stats['mean'] = visitor_stats['mean'] * 100
        visitor_stats['VisitorType'] = visitor_stats['VisitorType'].replace({
            'Returning_Visitor': 'Returning Visitors',
            'New_Visitor': 'New Visitors',
            'Other': 'Other'
        })
        
        fig_visitor = px.bar(
            visitor_stats,
            x='VisitorType',
            y='mean',
            text='mean',
            color='VisitorType',
            color_discrete_map={
                'New Visitors': '#6366F1',
                'Returning Visitors': '#94A3B8',
                'Other': '#CBD5E1'
            },
            labels={'mean': 'Purchase Rate (%)', 'VisitorType': 'Customer Segment'}
        )
        fig_visitor.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate='%{x}: <b>%{y:.1f}%</b> purchase rate<extra></extra>'
        )
        fig_visitor.update_layout(
            showlegend=False,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 32], showgrid=True, gridcolor='#F1F5F9', title='Conversion Rate (%)'),
            xaxis=dict(title='')
        )
        st.plotly_chart(fig_visitor, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> <b>New Visitors convert at 24.9%</b> compared to only <b>13.9% for Returning Visitors</b>! First-time visitors arrive with high intent, making first-impression welcomes and introductory discounts highly profitable.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='my-6'></div>", unsafe_allow_html=True)

    # Visual Row 2: Page Values & Monthly Seasonality
    r2_col1, r2_col2 = st.columns([1, 1])
    
    with r2_col1:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">3. The Power of "Page Value" (High-Intent Pages)</h4>
            <p class="text-xs text-slate-400 mb-3">Comparing average page value for purchasers vs non-purchasers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        pv_summary = df.groupby('Revenue')['PageValues'].mean().reset_index()
        pv_summary['Outcome'] = pv_summary['Revenue'].map({False: 'Did Not Purchase', True: 'Completed Purchase'})
        
        fig_pv = px.bar(
            pv_summary,
            x='Outcome',
            y='PageValues',
            text='PageValues',
            color='Outcome',
            color_discrete_map={
                'Did Not Purchase': '#CBD5E1',
                'Completed Purchase': '#10B981'
            },
            labels={'PageValues': 'Avg Page Value ($)'}
        )
        fig_pv.update_traces(
            texttemplate='$%{text:.2f}',
            textposition='outside',
            hovertemplate='%{x}: <b>$%{y:.2f}</b><extra></extra>'
        )
        fig_pv.update_layout(
            showlegend=False,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 35], showgrid=True, gridcolor='#F1F5F9', title='Avg Page Value ($)'),
            xaxis=dict(title='')
        )
        st.plotly_chart(fig_pv, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> Completed buyers visit pages with an average page value of <b>$27.26</b> vs only <b>$1.98</b> for non-buyers—a <b>13.8x difference</b>! Directing shoppers toward high-value pages is the quickest way to drive checkouts.
        </div>
        """, unsafe_allow_html=True)
        
    with r2_col2:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">4. Seasonality: Which Months Drive the Most Orders?</h4>
            <p class="text-xs text-slate-400 mb-3">Monthly completed order volume across the year.</p>
        </div>
        """, unsafe_allow_html=True)
        
        month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_sales = df[df['Revenue'] == True]['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
        monthly_sales.columns = ['Month', 'Completed Purchases']
        
        fig_month = px.bar(
            monthly_sales,
            x='Month',
            y='Completed Purchases',
            text='Completed Purchases',
            color='Completed Purchases',
            color_continuous_scale='Purples'
        )
        fig_month.update_traces(
            textposition='outside',
            hovertemplate='<b>%{x}</b>: %{y:,} purchases<extra></extra>'
        )
        fig_month.update_layout(
            coloraxis_showscale=False,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title='Purchases Count'),
            xaxis=dict(title='')
        )
        st.plotly_chart(fig_month, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> <b>November dominates with 756 purchases</b> (nearly 40% of all annual orders), followed by <b>May with 565 orders</b>. E-commerce teams must scale ad spend and optimize server infrastructure heavily prior to November.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='my-6'></div>", unsafe_allow_html=True)

    # Visual Row 3: Weekends vs Weekdays & Exit Rates
    r3_col1, r3_col2 = st.columns([1, 1])
    
    with r3_col1:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">5. Weekend vs. Weekday Shopping Behavior</h4>
            <p class="text-xs text-slate-400 mb-3">Do shoppers convert more frequently on Saturday & Sunday?</p>
        </div>
        """, unsafe_allow_html=True)
        
        wknd_summary = df.groupby('Weekend')['Revenue'].agg(['count', 'mean']).reset_index()
        wknd_summary['Rate'] = wknd_summary['mean'] * 100
        wknd_summary['Day'] = wknd_summary['Weekend'].map({False: 'Weekdays (Mon-Fri)', True: 'Weekends (Sat-Sun)'})
        
        fig_wknd = px.bar(
            wknd_summary,
            x='Day',
            y='Rate',
            text='Rate',
            color='Day',
            color_discrete_map={
                'Weekdays (Mon-Fri)': '#94A3B8',
                'Weekends (Sat-Sun)': '#3B82F6'
            }
        )
        fig_wknd.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            hovertemplate='%{x}: <b>%{y:.1f}%</b> conversion<extra></extra>'
        )
        fig_wknd.update_layout(
            showlegend=False,
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 24], showgrid=True, gridcolor='#F1F5F9', title='Conversion Rate (%)'),
            xaxis=dict(title='')
        )
        st.plotly_chart(fig_wknd, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> Weekend shoppers show a slightly higher conversion rate (<b>17.4%</b>) compared to weekdays (<b>14.9%</b>). Weekend marketing promotions (e.g. "Weekend Flash Sales") align well with consumer availability.
        </div>
        """, unsafe_allow_html=True)
        
    with r3_col2:
        st.markdown("""
        <div class="modern-card">
            <h4 class="font-bold text-slate-800 text-base mb-1">6. Friction Point: The Impact of Exit Rates</h4>
            <p class="text-xs text-slate-400 mb-3">Comparing average exit rates for buyers vs non-buyers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        exit_summary = df.groupby('Revenue')['ExitRates'].mean().reset_index()
        exit_summary['ExitPct'] = exit_summary['ExitRates'] * 100
        exit_summary['Outcome'] = exit_summary['Revenue'].map({False: 'Did Not Purchase', True: 'Completed Purchase'})
        
        fig_exit = px.bar(
            exit_summary,
            x='Outcome',
            y='ExitPct',
            text='ExitPct',
            color='Outcome',
            color_discrete_map={
                'Did Not Purchase': '#F43F5E',
                'Completed Purchase': '#10B981'
            }
        )
        fig_exit.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
            hovertemplate='%{x}: <b>%{y:.2f}%</b> exit rate<extra></extra>'
        )
        fig_exit.update_layout(
            showlegend=False,
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 7], showgrid=True, gridcolor='#F1F5F9', title='Average Exit Rate (%)'),
            xaxis=dict(title='')
        )
        st.plotly_chart(fig_exit, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <b>📌 Business Takeaway:</b> Non-purchasing sessions average an exit rate of <b>4.74%</b>, which is <b>more than double</b> that of buyers (<b>1.94%</b>). Pages with above-average exit rates should be tested for friction, slow loading, or unclear CTA buttons.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: PURCHASE PREDICTOR (INTERACTIVE ML)
# ==========================================
elif page == "🔮 Purchase Predictor":
    st.markdown("""
    <div class="mb-6">
        <span class="badge badge-indigo mb-2">MACHINE LEARNING INFERENCE</span>
        <h1 class="text-2xl md:text-3xl font-extrabold text-slate-800 tracking-tight">Real-Time Purchase Intent Predictor</h1>
        <p class="text-slate-500 text-sm mt-1">
            Simulate a customer's browsing session and utilize the trained <b>XGBoost Classifier</b> to estimate whether they are likely to complete a purchase.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("ML Model Pipeline ('XGBoost pipeline final project.pkl') could not be loaded. Please ensure the model file is present.")
    else:
        with st.form("prediction_form"):
            st.markdown("<p class='font-bold text-slate-700 text-sm mb-3'>1. Browsing Behavior & Page Interactions</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                product_related = st.number_input("Product Pages Visited", min_value=0, max_value=700, value=15, step=1)
                product_duration = st.number_input("Time on Product Pages (Sec)", min_value=0.0, max_value=60000.0, value=450.0, step=10.0)
            with col2:
                admin_pages = st.number_input("Account/Admin Pages Visited", min_value=0, max_value=30, value=1, step=1)
                admin_duration = st.number_input("Time on Admin Pages (Sec)", min_value=0.0, max_value=5000.0, value=30.0, step=5.0)
            with col3:
                info_pages = st.number_input("Info/FAQ Pages Visited", min_value=0, max_value=30, value=0, step=1)
                info_duration = st.number_input("Time on Info Pages (Sec)", min_value=0.0, max_value=5000.0, value=0.0, step=5.0)
                
            st.markdown("<hr class='my-4 border-slate-200'>", unsafe_allow_html=True)
            st.markdown("<p class='font-bold text-slate-700 text-sm mb-3'>2. Engagement & Site Metrics</p>", unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            with col4:
                page_values = st.number_input("Google Analytics Page Value ($)", min_value=0.0, max_value=400.0, value=25.0, step=1.0, help="High page value indicates visiting pages proven to generate revenue.")
            with col5:
                exit_rates = st.slider("Exit Rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100.0
            with col6:
                bounce_rates = st.slider("Bounce Rate (%)", min_value=0.0, max_value=20.0, value=0.5, step=0.1) / 100.0
                
            st.markdown("<hr class='my-4 border-slate-200'>", unsafe_allow_html=True)
            st.markdown("<p class='font-bold text-slate-700 text-sm mb-3'>3. Visitor Context & Timing</p>", unsafe_allow_html=True)
            col7, col8, col9 = st.columns(3)
            with col7:
                visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
                month = st.selectbox("Month of Session", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8) # Default Nov
            with col8:
                weekend = st.selectbox("Is Weekend?", [False, True], index=0)
                special_day = st.slider("Proximity to Special Day / Holiday", min_value=0.0, max_value=1.0, value=0.0, step=0.2)
            with col9:
                traffic_type = st.number_input("Traffic Source Code (1-20)", min_value=1, max_value=20, value=2, step=1)
                os_type = st.number_input("Operating System Code (1-8)", min_value=1, max_value=8, value=2, step=1)
                browser_code = st.number_input("Browser Code (1-13)", min_value=1, max_value=13, value=2, step=1)
                region_code = st.number_input("Region Code (1-9)", min_value=1, max_value=9, value=1, step=1)
                
            submitted = st.form_submit_button("🔮 Predict Purchase Likelihood", use_container_width=True)
            
        if submitted:
            # Construct DataFrame matching pipeline input exactly
            input_dict = {
                'Administrative': [admin_pages],
                'Administrative_Duration': [admin_duration],
                'Informational': [info_pages],
                'Informational_Duration': [info_duration],
                'ProductRelated': [product_related],
                'ProductRelated_Duration': [product_duration],
                'BounceRates': [bounce_rates],
                'ExitRates': [exit_rates],
                'PageValues': [page_values],
                'SpecialDay': [special_day],
                'Month': [month],
                'OperatingSystems': [os_type],
                'Browser': [browser_code],
                'Region': [region_code],
                'TrafficType': [traffic_type],
                'VisitorType': [visitor_type],
                'Weekend': [weekend]
            }
            input_df = pd.DataFrame(input_dict)
            
            try:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                purchase_prob = proba[1] * 100
                
                st.markdown("<hr class='my-6 border-slate-200'>", unsafe_allow_html=True)
                
                if prediction == 1 or purchase_prob >= 50.0:
                    st.markdown(f"""
                    <div class="bg-emerald-50 border-2 border-emerald-500 rounded-2xl p-6 shadow-md">
                        <div class="flex items-center space-x-3 mb-2">
                            <span class="text-3xl">🎉</span>
                            <div>
                                <h3 class="text-xl font-black text-emerald-800">High Purchase Likelihood: Likely to Buy!</h3>
                                <p class="text-emerald-700 text-xs font-semibold">The XGBoost model predicts this session will convert into a completed sale.</p>
                            </div>
                        </div>
                        <div class="my-4">
                            <div class="flex justify-between text-xs font-bold text-emerald-800 mb-1">
                                <span>Estimated Conversion Probability</span>
                                <span>{purchase_prob:.1f}%</span>
                            </div>
                            <div class="w-full bg-emerald-200 rounded-full h-3">
                                <div class="bg-emerald-600 h-3 rounded-full" style="width: {min(100, purchase_prob)}%"></div>
                            </div>
                        </div>
                        <div class="bg-white/80 p-3 rounded-xl text-xs text-emerald-900 border border-emerald-200 space-y-1">
                            <p class="font-bold">💼 Recommended Marketing Action:</p>
                            <p>• Avoid intrusive popups or disruptive discounts—this visitor is already qualified and proceeding to checkout naturally.</p>
                            <p>• Offer seamless 1-click checkout or free shipping reassurance at cart to secure the transaction.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bg-amber-50 border-2 border-amber-400 rounded-2xl p-6 shadow-md">
                        <div class="flex items-center space-x-3 mb-2">
                            <span class="text-3xl">👀</span>
                            <div>
                                <h3 class="text-xl font-black text-amber-900">Low Purchase Likelihood: Browsing Only</h3>
                                <p class="text-amber-700 text-xs font-semibold">The XGBoost model predicts this visitor will likely drop off without buying.</p>
                            </div>
                        </div>
                        <div class="my-4">
                            <div class="flex justify-between text-xs font-bold text-amber-900 mb-1">
                                <span>Estimated Conversion Probability</span>
                                <span>{purchase_prob:.1f}%</span>
                            </div>
                            <div class="w-full bg-amber-200 rounded-full h-3">
                                <div class="bg-amber-500 h-3 rounded-full" style="width: {max(5, purchase_prob)}%"></div>
                            </div>
                        </div>
                        <div class="bg-white/80 p-3 rounded-xl text-xs text-amber-900 border border-amber-200 space-y-1">
                            <p class="font-bold">💼 Recommended Retention Action:</p>
                            <p>• Trigger an exit-intent discount popup (e.g., 10% off with coupon code).</p>
                            <p>• Show personalized product recommendations or customer reviews on their currently viewed category.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction calculation failed: {e}")

