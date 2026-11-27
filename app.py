                        )
                        fig_hist.add_vline(
                            x=threshold,
                            line_dash="dash",
                            line_color="#F5A623",
                            annotation_text="Selected threshold",
                        )
                        fig_hist.update_layout(
                            paper_bgcolor="#08111F",
                            plot_bgcolor="#111E30",
                            font_color="#EAF2FA",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                section_heading("📋", "Transaction-level results")
                display_columns = [
                    "leakage_probability",
                    "leakage_prediction",
                    "leakage_status",
                    "risk_level",
                ]
                if regressor is not None:
                    display_columns += [
                        "estimated_leakage_amount_if_leakage",
                        "expected_revenue_exposure",
                    ]

                display = pd.concat(
                    [
                        uploaded_data.reset_index(drop=True),
                        batch_results[display_columns].reset_index(drop=True),
                    ],
                    axis=1,
                )

                st.dataframe(display, use_container_width=True, height=430)

                st.download_button(
                    "⬇️ Download prediction results",
                    data=display.to_csv(index=False).encode("utf-8"),
                    file_name="revenueguard_ai_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as exc:
            st.error(f"Unable to process the uploaded CSV: {exc}")
            st.caption(
                "Check that the file is a valid CSV and that its predictor "
                "columns contain appropriate values."
            )

elif page == "🧠 About the model":
    section_heading(
        "🧠",
        "Model information",
        "Understand the model's purpose, inputs and current deployment status.",
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">CLASSIFICATION ENGINE</div>
              <h3>Logistic Regression</h3>
              <p style="color:#B9C8D8;">
                The classification pipeline estimates the probability that a
                transaction belongs to the revenue leakage class.
              </p>
              <hr>
              <p><strong>Class 1:</strong> Revenue leakage</p>
              <p><strong>Class 0:</strong> No revenue leakage</p>
              <p><strong>Decision threshold:</strong> Adjustable from the sidebar</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="rg-card">
              <div class="rg-section-label">PREDICTOR INPUTS</div>
              <h3>20 transaction features</h3>
              <p style="color:#B9C8D8;">
                Inputs cover transaction value, customer history, sales
                channels, payment terms, discounts, billing, operations and
                contract controls.
              </p>
              <hr>
              <p>🧾 Transaction & customer</p>
              <p>💳 Payment & billing</p>
              <p>⚙️ Operational controls</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 🔧 Deployment details")
    st.write(f"**Classifier artifact:** `{classifier_path}`")
    st.write(
        f"**Regression artifact:** "
        f"`{regressor_path if regressor_path else 'Not available yet'}`"
    )
    st.write(f"**Current decision threshold:** `{threshold:.2f}`")
    st.write(f"**Required predictor count:** `{len(FEATURE_COLUMNS)}`")

    with st.expander("View all 20 required predictor columns"):
        st.code("\n".join(FEATURE_COLUMNS), language="text")

    st.warning(
        "The 0.10 threshold is only a starting value if it was saved from the "
        "notebook example. Select the deployment threshold using validation "
        "data and business costs, then evaluate it on an untouched test set."
    )

    st.info(
        "Model predictions are risk indicators, not proof that leakage has "
        "occurred. Validate the application with representative real business "
        "data before using it for operational decisions."
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.caption(
        "🛡️ RevenueGuard AI | Predictive Revenue Leakage Intelligence"
    )
with footer_right:
    st.caption("Built for better revenue protection decisions.")
