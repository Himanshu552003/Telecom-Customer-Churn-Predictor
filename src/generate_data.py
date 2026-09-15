"""
Generate a realistic synthetic telecom customer churn dataset.

The dataset mimics the well-known IBM Telco Customer Churn schema, but is
fully synthetic. Churn probability is driven by a mix of known real-world
risk factors (month-to-month contracts, short tenure, high monthly charges,
lack of add-on/security services, electronic check payment, no tech support)
combined with random noise, so the signal is realistic but not trivially
separable.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7043  # matches classic telco dataset scale

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n=N_CUSTOMERS):
    customer_id = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
                   for _ in range(n)]

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])

    # Dependents more likely if has partner
    dependents = np.where(
        partner == "Yes",
        rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55]),
        rng.choice(["Yes", "No"], size=n, p=[0.15, 0.85]),
    )

    # Tenure in months: skew toward shorter tenures, cap at 72 months
    tenure = rng.gamma(shape=2.0, scale=15, size=n).astype(int)
    tenure = np.clip(tenure, 0, 72)

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])

    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def addon_service(p_yes=0.35):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes]),
        )

    online_security = addon_service(0.29)
    online_backup = addon_service(0.35)
    device_protection = addon_service(0.34)
    tech_support = addon_service(0.29)
    streaming_tv = addon_service(0.38)
    streaming_movies = addon_service(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )

    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])

    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- Monthly charges: base + add-ons ---
    base_charge = rng.normal(20, 3, size=n)
    internet_charge = np.select(
        [internet_service == "DSL", internet_service == "Fiber optic", internet_service == "No"],
        [rng.normal(25, 4, size=n), rng.normal(45, 6, size=n), 0],
    )
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    addon_charge = addon_count * rng.normal(5, 1, size=n)
    phone_charge = np.where(phone_service == "Yes", rng.normal(8, 2, size=n), 0)

    monthly_charges = np.clip(base_charge + internet_charge + addon_charge + phone_charge, 18, 120)
    monthly_charges = np.round(monthly_charges, 2)

    # Total charges roughly tenure * monthly_charges with some noise;
    # brand-new customers (tenure=0) get a blank string like the real dataset.
    total_charges = np.round(
        monthly_charges * tenure + rng.normal(0, 20, size=n).clip(-50, 50), 2
    )
    total_charges = np.clip(total_charges, 0, None)
    total_charges_str = total_charges.astype(str)
    total_charges_str[tenure == 0] = " "  # mimic real-world blank TotalCharges

    # --- Churn probability model (logistic-style score + noise) ---
    score = np.zeros(n)
    score += np.where(contract == "Month-to-month", 1.1, 0)
    score += np.where(contract == "One year", 0.15, 0)
    score += np.where(contract == "Two year", -1.2, 0)

    score += -0.045 * tenure  # longer tenure -> less churn
    score += 0.018 * (monthly_charges - 65)  # higher charges -> more churn

    score += np.where(internet_service == "Fiber optic", 0.45, 0)
    score += np.where(internet_service == "No", -0.35, 0)

    score += np.where(online_security == "No", 0.25, 0)
    score += np.where(tech_support == "No", 0.25, 0)

    score += np.where(payment_method == "Electronic check", 0.35, 0)
    score += np.where(paperless_billing == "Yes", 0.15, 0)

    score += np.where(senior_citizen == 1, 0.2, 0)
    score += np.where(partner == "No", 0.1, 0)
    score += np.where(dependents == "No", 0.08, 0)

    # noise term keeps it from being trivially separable
    score += rng.normal(0, 0.9, size=n)

    # calibrate intercept to hit ~26-28% churn rate
    intercept = -1.35
    prob_churn = 1 / (1 + np.exp(-(score + intercept)))
    churn = rng.binomial(1, prob_churn)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn_label,
        }
    )

    # ensure unique customer IDs
    df["customerID"] = df["customerID"] + "-" + pd.Series(np.arange(n)).astype(str)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/telecom_churn.csv"
    df.to_csv(out_path, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Overall churn rate: {churn_rate:.2%}")
    print(df.head())
