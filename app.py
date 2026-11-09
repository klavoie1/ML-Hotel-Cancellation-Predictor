"""
Booking Cancellation Predictor dashboard.

Start it with:
    streamlit run app.py
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import database
import security
import train_model

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
LOG_DIR = Path("logs")
MAX_ATTEMPTS = 5

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_DIR / "app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="Booking Cancellation Predictor", layout="wide")


# ---------- Security: login gate ----------
def login_screen():
    st.title("Booking Cancellation Predictor")
    attempts = st.session_state.get("attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        st.error("Too many failed sign-in attempts. Restart the app to try again.")
        st.stop()
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if security.check_login(username.strip(), password):
                st.session_state["user"] = username.strip()
                logging.info("Login succeeded for user=%s", username.strip())
                st.rerun()
            else:
                st.session_state["attempts"] = attempts + 1
                logging.warning("Login failed for user=%s", username.strip())
                st.error("Incorrect username or password.")
    st.stop()


if "user" not in st.session_state:
    login_screen()


@st.cache_data
def load_data():
    return pd.read_csv(Path("data/hotel_bookings_clean.csv"), parse_dates=["arrival_date"])


@st.cache_resource
def load_model():
    model = joblib.load(Path("models/model.joblib"))
    metrics = json.loads(Path("models/metrics.json").read_text())
    return model, metrics


df = load_data()
model, metrics = load_model()


def choose(column, label, where):
    """Dropdown of a column's values, starting on the most common one."""
    options = sorted(df[column].unique())
    return where.selectbox(label, options, index=options.index(df[column].mode()[0]))


st.sidebar.write(f"Signed in as **{st.session_state['user']}**")
if st.sidebar.button("Sign out"):
    logging.info("Logout for user=%s", st.session_state["user"])
    st.session_state.clear()
    st.rerun()

st.title("Booking Cancellation Predictor")
page = st.sidebar.radio("Go to", ["Dashboard", "Check a booking", "Model accuracy", "Monitoring"])

# ---------- Dashboard: descriptive analysis ----------
if page == "Dashboard":
    hotels = st.sidebar.multiselect("Hotel", sorted(df["hotel"].unique()),
                                    default=sorted(df["hotel"].unique()))
    segments = st.sidebar.multiselect("Market segment", sorted(df["market_segment"].unique()),
                                      default=sorted(df["market_segment"].unique()))
    view = df[df["hotel"].isin(hotels) & df["market_segment"].isin(segments)]
    if view.empty:
        st.warning("No bookings match these filters.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Bookings", f"{len(view):,}")
    b.metric("Cancellation rate", f"{view['is_canceled'].mean():.1%}")
    c.metric("Average lead time", f"{view['lead_time'].mean():.0f} days")

    by_month = (view.groupby([view["arrival_date"].dt.to_period("M").dt.to_timestamp(), "hotel"])
                ["is_canceled"].mean().reset_index(name="rate"))
    fig1 = px.line(by_month, x="arrival_date", y="rate", color="hotel", markers=True,
                   labels={"arrival_date": "Arrival month", "rate": "Cancellation rate"},
                   title="Cancellation rate by arrival month")
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, width="stretch")

    left, right = st.columns(2)
    by_deposit = view.groupby("deposit_type")["is_canceled"].mean().reset_index(name="rate")
    fig2 = px.bar(by_deposit, x="deposit_type", y="rate",
                  labels={"deposit_type": "Deposit type", "rate": "Cancellation rate"},
                  title="Cancellation rate by deposit type")
    fig2.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig2, width="stretch")

    outcome = view["is_canceled"].map({0: "Kept", 1: "Canceled"})
    fig3 = px.box(view.assign(outcome=outcome), x="outcome", y="lead_time", points=False,
                  labels={"outcome": "Booking outcome", "lead_time": "Lead time (days)"},
                  title="Lead time for kept vs. canceled bookings")
    right.plotly_chart(fig3, width="stretch")

    with st.expander("Explore the data"):
        st.write("Summary statistics for the filtered bookings")
        st.dataframe(view.select_dtypes("number").describe().T.round(2), width="stretch")
        st.write("Correlation of each numeric booking detail with cancellation "
                 "(positive means more likely to cancel)")
        corr = (view.select_dtypes("number").corr()["is_canceled"].drop("is_canceled")
                .sort_values(ascending=False).round(3).rename("correlation"))
        st.dataframe(corr, width="stretch")
        st.write("First 100 filtered bookings")
        st.dataframe(view.head(100), width="stretch")

# ---------- Check a booking: predictive method and decision support ----------
elif page == "Check a booking":
    st.subheader("Check a booking's cancellation risk")
    with st.form("booking"):
        c1, c2, c3 = st.columns(3)
        hotel = choose("hotel", "Hotel", c1)
        month = c1.selectbox("Arrival month", MONTHS, index=6)
        lead = c1.number_input("Lead time (days before arrival)", 0, 750, 30)
        weekend = c1.number_input("Weekend nights", 0, 20, 1)
        week = c1.number_input("Week nights", 0, 50, 2)
        prev_cancel = c1.number_input("Previous cancellations", 0, 30, 0)
        prev_kept = c1.number_input("Previous completed stays", 0, 80, 0)
        adults = c2.number_input("Adults", 1, 10, 2)
        children = c2.number_input("Children", 0, 10, 0)
        babies = c2.number_input("Babies", 0, 10, 0)
        meal = choose("meal", "Meal plan", c2)
        room = choose("reserved_room_type", "Room type", c2)
        adr = c2.number_input("Average daily rate", 0.0, 500.0, 100.0)
        parking = c2.number_input("Parking spaces requested", 0, 3, 0)
        deposit = choose("deposit_type", "Deposit type", c3)
        segment = choose("market_segment", "Market segment", c3)
        channel = choose("distribution_channel", "Booking channel", c3)
        customer = choose("customer_type", "Customer type", c3)
        requests = c3.number_input("Special requests", 0, 5, 0)
        agent = c3.checkbox("Booked through a travel agent")
        repeat = c3.checkbox("Repeat guest")
        submitted = st.form_submit_button("Check risk")

    if submitted:
        if weekend + week == 0:  # input validation beyond the field limits
            st.error("A booking must include at least one night.")
            st.stop()
        booking = pd.DataFrame([{
            "lead_time": lead, "stays_in_weekend_nights": weekend, "stays_in_week_nights": week,
            "adults": adults, "children": children, "babies": babies,
            "is_repeated_guest": int(repeat), "previous_cancellations": prev_cancel,
            "previous_bookings_not_canceled": prev_kept, "adr": adr,
            "required_car_parking_spaces": parking, "total_of_special_requests": requests,
            "has_agent": int(agent), "hotel": hotel, "arrival_date_month": month, "meal": meal,
            "market_segment": segment, "distribution_channel": channel,
            "reserved_room_type": room, "deposit_type": deposit, "customer_type": customer}])
        prob = model.predict_proba(booking)[0, 1]
        risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"
        st.metric("Chance of cancellation", f"{prob:.0%}")

        if risk == "High":
            st.error("High risk: consider requesting a deposit and counting this room "
                     "toward overbooking.")
        elif risk == "Medium":
            st.warning("Medium risk: consider sending a confirmation request closer to arrival.")
        else:
            st.success("Low risk: no action needed.")

        # Monitoring: record the prediction (booking details only, no personal data).
        record = booking.assign(probability=round(prob, 4), risk=risk,
                                user=st.session_state["user"],
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with database.connect() as con:
            record.to_sql("predictions", con, if_exists="append", index=False)
        logging.info("Prediction by user=%s probability=%.3f risk=%s",
                     st.session_state["user"], prob, risk)

# ---------- Model accuracy and hypotheses ----------
elif page == "Model accuracy":
    test = metrics["test"]
    st.subheader(f"{metrics['chosen_model']} performance on {metrics['test_rows']:,} test bookings")
    cols = st.columns(5)

    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.replace("_", " ").upper() if key == "roc_auc" else key.title(),
                   f"{test[key]:.1%}" if key != "roc_auc" else f"{test[key]:.3f}")

    st.caption(f"Logistic regression baseline accuracy: "
               f"{metrics['baseline_logistic_regression']['accuracy']:.1%}. "
               f"Always guessing 'not canceled': {metrics['always_not_canceled_accuracy']:.1%}.")

    left, right = st.columns(2)

    cm = pd.DataFrame(test["confusion_matrix"], index=["Actually kept", "Actually canceled"],
                      columns=["Predicted kept", "Predicted canceled"])

    fig4 = px.imshow(cm, text_auto=",d", color_continuous_scale="Blues", title="Confusion matrix")

    left.plotly_chart(fig4, width="stretch")

    if metrics["top_features"]:
        top = pd.DataFrame(metrics["top_features"], columns=["feature", "importance"])
        fig5 = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h",
                      title="Top 10 factors the model relies on")
        right.plotly_chart(fig5, width="stretch")

    st.subheader("Hypothesis results")
    st.dataframe(pd.DataFrame(metrics["hypotheses"]).set_index("id"), width="stretch")
    st.subheader("Tuning results (5-fold cross-validation)")
    st.dataframe(pd.DataFrame(metrics["grid_results"]), width="stretch", hide_index=True)

# ---------- Monitoring and maintenance ----------
else:
    st.subheader("Monitoring and maintenance")
    a, b, c = st.columns(3)
    a.metric("Model trained", metrics["trained_at"])
    b.metric("Test accuracy", f"{metrics['test']['accuracy']:.1%}")
    c.metric("Training bookings", f"{metrics['train_rows']:,}")

    if metrics["test"]["accuracy"] < 0.80:
        st.error("Accuracy is below the 80% target. Retrain the model with newer data.")

    with database.connect() as con:
        has_log = con.execute("SELECT name FROM sqlite_master "
                              "WHERE type = 'table' AND name = 'predictions'").fetchone()
        log = pd.read_sql("SELECT * FROM predictions", con) if has_log else pd.DataFrame()
    if not log.empty:
        st.write(f"{len(log):,} predictions logged. "
                 f"Average predicted risk: {log['probability'].mean():.0%}.")
        st.dataframe(log[["timestamp", "user", "hotel", "lead_time", "deposit_type",
                          "probability", "risk"]].tail(20).iloc[::-1],
                     width="stretch", hide_index=True)
    else:
        st.info("No predictions logged yet.")

    log_file = LOG_DIR / "app.log"

    if log_file.exists():
        with st.expander("Recent activity log"):
            st.code("".join(log_file.read_text().splitlines(keepends=True)[-20:]))

    st.write("**Retrain the model** with the data in `data/hotel_bookings_raw.csv`. "
             "Replace that file with a newer export first to train on new bookings.")

    if st.button("Retrain model"):
        with st.spinner("Retraining. This takes about a minute..."):
            train_model.main()
        logging.info("Model retrained by user=%s", st.session_state["user"])
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Model retrained. Reloading...")
        st.rerun()
