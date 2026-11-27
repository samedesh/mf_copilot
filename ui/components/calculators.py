"""Interactive financial calculators matching app.tools.calculator formulas."""
from __future__ import annotations
import pandas as pd
import streamlit as st

from app.tools.calculator import run as run_calc


def render_calculators():
    """Render financial calculators workbench."""
    st.markdown("## 🧮 Financial Calculators Workbench")
    st.markdown("Deterministic financial mathematics matching the agent's verified calculation tool.")

    calc_tab1, calc_tab2, calc_tab3 = st.tabs([
        "📈 SIP Future Value",
        "💸 Expense Ratio (TER) Impact",
        "📊 CAGR & % Change",
    ])

    with calc_tab1:
        st.markdown("### Systematic Investment Plan (SIP) Compounding")
        st.caption("Calculates maturity value of monthly investments made at the start of each month.")

        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=500.0)
        with col2:
            annual_rate = st.number_input("Expected Annual Return (%)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
        with col3:
            years = st.slider("Investment Horizon (Years)", min_value=1, max_value=40, value=10)

        # Run calculation using app.tools.calculator
        res = run_calc("sip_fv", {"monthly": monthly_sip, "annual_rate_pct": annual_rate, "years": years})
        fv = res["value"]
        total_invested = monthly_sip * years * 12
        total_gain = fv - total_invested

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Invested", f"₹ {total_invested:,.0f}")
        with m2:
            st.metric("Estimated Wealth (FV)", f"₹ {fv:,.0f}")
        with m3:
            st.metric("Estimated Returns", f"₹ {total_gain:,.0f}", delta=f"{((fv/total_invested)-1)*100:.1f}%")

        # Year by year table and chart
        progression = []
        for y in range(1, years + 1):
            y_fv = run_calc("sip_fv", {"monthly": monthly_sip, "annual_rate_pct": annual_rate, "years": y})["value"]
            y_invested = monthly_sip * y * 12
            progression.append({
                "Year": y,
                "Invested Capital": round(y_invested, 2),
                "Total Value": round(y_fv, 2),
                "Wealth Gain": round(y_fv - y_invested, 2),
            })

        df_sip = pd.DataFrame(progression)
        st.markdown("#### Growth Over Time")
        st.line_chart(df_sip.set_index("Year")[["Invested Capital", "Total Value"]])

        with st.expander("Detailed Year-by-Year Schedule"):
            st.dataframe(df_sip.style.format({"Invested Capital": "₹ {:,.0f}", "Total Value": "₹ {:,.0f}", "Wealth Gain": "₹ {:,.0f}"}), use_container_width=True)

    with calc_tab2:
        st.markdown("### Expense Ratio (TER) Drag Comparison")
        st.caption("Compare how distributor commissions and higher TER in Regular plans erode long-term compounding vs Direct plans.")

        col1, col2, col3 = st.columns(3)
        with col1:
            lumpsum = st.number_input("Investment Amount (₹)", min_value=10_000.0, max_value=100_000_000.0, value=500_000.0, step=50_000.0)
        with col2:
            gross_return = st.number_input("Expected Gross Portfolio Return (%)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
        with col3:
            comp_years = st.slider("Holding Period (Years)", min_value=1, max_value=35, value=15)

        col_ter1, col_ter2 = st.columns(2)
        with col_ter1:
            ter_direct = st.number_input("Direct Plan Expense Ratio (%)", min_value=0.0, max_value=5.0, value=0.75, step=0.05)
        with col_ter2:
            ter_regular = st.number_input("Regular Plan Expense Ratio (%)", min_value=0.0, max_value=5.0, value=1.75, step=0.05)

        # Calculate using app.tools.calculator
        cost_direct = run_calc("expense_cost", {"amount": lumpsum, "expense_pct": ter_direct, "gross_return_pct": gross_return, "years": comp_years})["value"]
        cost_regular = run_calc("expense_cost", {"amount": lumpsum, "expense_pct": ter_regular, "gross_return_pct": gross_return, "years": comp_years})["value"]
        gross_fv = lumpsum * (1 + gross_return / 100) ** comp_years
        direct_fv = gross_fv - cost_direct
        regular_fv = gross_fv - cost_regular
        lost_to_commission = direct_fv - regular_fv

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Direct Plan Final Wealth", f"₹ {direct_fv:,.0f}")
        with m2:
            st.metric("Regular Plan Final Wealth", f"₹ {regular_fv:,.0f}")
        with m3:
            st.metric("Lost to Commission / TER Drag", f"₹ {lost_to_commission:,.0f}", delta=f"-{lost_to_commission:,.0f}", delta_color="inverse")

        # Multi-year comparison dataframe
        schedule = []
        for y in range(1, comp_years + 1):
            y_gross = lumpsum * (1 + gross_return / 100) ** y
            y_dir_cost = run_calc("expense_cost", {"amount": lumpsum, "expense_pct": ter_direct, "gross_return_pct": gross_return, "years": y})["value"]
            y_reg_cost = run_calc("expense_cost", {"amount": lumpsum, "expense_pct": ter_regular, "gross_return_pct": gross_return, "years": y})["value"]
            y_dir_fv = y_gross - y_dir_cost
            y_reg_fv = y_gross - y_reg_cost
            schedule.append({
                "Year": y,
                "Direct Plan Wealth": round(y_dir_fv, 2),
                "Regular Plan Wealth": round(y_reg_fv, 2),
                "Wealth Difference": round(y_dir_fv - y_reg_fv, 2),
            })
        df_ter = pd.DataFrame(schedule)
        st.markdown("#### Wealth Gap Evolution Over Time")
        st.line_chart(df_ter.set_index("Year")[["Direct Plan Wealth", "Regular Plan Wealth"]])

    with calc_tab3:
        st.markdown("### CAGR & Percentage Change")
        col_cagr, col_pct = st.columns(2)

        with col_cagr:
            st.markdown("#### Compound Annual Growth Rate (CAGR)")
            start_val = st.number_input("Beginning Value (₹)", min_value=0.01, value=100.0, step=10.0)
            end_val = st.number_input("Ending Value (₹)", min_value=0.01, value=250.0, step=10.0)
            cagr_years = st.number_input("Period (Years)", min_value=0.1, value=5.0, step=0.5)

            if st.button("Compute CAGR"):
                try:
                    cagr_res = run_calc("cagr", {"start": start_val, "end": end_val, "years": cagr_years})
                    st.success(f"**CAGR:** {cagr_res['value']:.2f}% per annum")
                    st.caption(f"Formula: `{cagr_res['formula']}`")
                except Exception as e:
                    st.error(str(e))

        with col_pct:
            st.markdown("#### Percentage Change")
            old_val = st.number_input("Base Value", value=100.0, step=5.0)
            new_val = st.number_input("New Value", value=120.0, step=5.0)

            if st.button("Compute % Change"):
                try:
                    pct_res = run_calc("pct_change", {"old": old_val, "new": new_val})
                    st.success(f"**Change:** {pct_res['value']:.2f}%")
                    st.caption(f"Formula: `{pct_res['formula']}`")
                except Exception as e:
                    st.error(str(e))
