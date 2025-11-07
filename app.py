
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Bank — Loan Propensity (Marketing)", layout="wide")

st.title("Universal Bank — Personal Loan Propensity Dashboard")
st.markdown("Head of Marketing dashboard: explore data, train models, and predict on new customers.")

# --- Helpers ---
@st.cache_data
def load_sample_data():
    # Try to load uploaded dataset file path used earlier in the notebook if present, else create placeholder instructions.
    try:
        df = pd.read_csv('/mnt/data/UniversalBank.csv')
    except Exception:
        # create sample minimal dataset for UI demo (user should upload real data)
        n = 1000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'ID': np.arange(1,n+1),
            'Age': rng.integers(22,65,n),
            'Experience': rng.integers(0,40,n),
            'Income': rng.integers(10,200,n),
            'ZIP Code': rng.integers(10000,99999,n),
            'Family': rng.integers(1,4,n),
            'CCAvg': np.round(rng.random(n)*10,2),
            'Education': rng.choice([1,2,3],n, p=[0.6,0.3,0.1]),
            'Mortgage': rng.integers(0,400,n),
            'Securities Account': rng.choice([0,1],n,p=[0.9,0.1]),
            'CD Account': rng.choice([0,1],n,p=[0.95,0.05]),
            'Online': rng.choice([0,1],n,p=[0.7,0.3]),
            'CreditCard': rng.choice([0,1],n,p=[0.7,0.3]),
        })
        # create a synthetic target correlated with Income, Education and CCAvg
        logits = (df['Income']*0.02 + df['CCAvg']*0.5 + (df['Education']-1)*1.0) - 4
        probs = 1/(1+np.exp(-logits))
        df['Personal Loan'] = (np.random.rand(n) < probs).astype(int)
    return df

def preprocess(df):
    df = df.copy()
    # standardize column names
    df.columns = [c.strip() for c in df.columns]
    # drop ID and zip if present
    for c in ['ID','Zip Code','ZIP Code','Zip','Zipcode']:
        if c in df.columns:
            try:
                df = df.drop(columns=[c])
            except Exception:
                pass
    # ensure target exists
    target_candidates = [c for c in df.columns if c.strip().lower().replace(" ","") in ['personalloan','personal_loan']]
    if not target_candidates:
        raise ValueError("No 'Personal Loan' target column found. Please ensure your file has a 'Personal Loan' column (1/0).")
    target = target_candidates[0]
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

@st.cache_data
def train_models(X, y, random_state=42):
    # Standardize numeric
    cont = [c for c in X.columns if c.lower() in ['age','experience','income','ccavg','mortgage']]
    scaler = StandardScaler()
    Xs = X.copy()
    if cont:
        Xs[cont] = scaler.fit_transform(X[cont])
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, stratify=y, test_size=0.30, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if hasattr(model,'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:,1]
        else:
            y_test_proba = y_test_pred
        results[name] = {
            'model': model,
            'cv_mean': float(np.round(cv_acc.mean(),4)),
            'cv_std': float(np.round(cv_acc.std(),4)),
            'train_acc': float(np.round(accuracy_score(y_train, y_train_pred),4)),
            'test_acc': float(np.round(accuracy_score(y_test, y_test_pred),4)),
            'precision': float(np.round(precision_score(y_test, y_test_pred, zero_division=0),4)),
            'recall': float(np.round(recall_score(y_test, y_test_pred, zero_division=0),4)),
            'f1': float(np.round(f1_score(y_test, y_test_pred, zero_division=0),4)),
            'auc': float(np.round(roc_auc_score(y_test, y_test_proba),4)),
            'y_test': y_test,
            'y_test_proba': y_test_proba,
            'y_test_pred': y_test_pred,
            'X_test': X_test,
            'feature_names': X.columns.tolist()
        }
    return results, scaler, Xs, X_train, X_test, y_train, y_test

def metrics_table(results):
    rows = []
    for k,v in results.items():
        rows.append({
            'Algorithm': k,
            'CV Accuracy Mean (5-fold)': v['cv_mean'],
            'CV Accuracy Std (5-fold)': v['cv_std'],
            'Training Accuracy': v['train_acc'],
            'Testing Accuracy': v['test_acc'],
            'Precision': v['precision'],
            'Recall': v['recall'],
            'F1-Score': v['f1'],
            'AUC': v['auc']
        })
    return pd.DataFrame(rows).set_index('Algorithm')

def plot_roc_all(results):
    fig = go.Figure()
    for name,v in results.items():
        fpr, tpr, _ = roc_curve(v['y_test'], v['y_test_proba'])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={v['auc']})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', showlegend=False, line=dict(dash='dash')))
    fig.update_layout(title="ROC Curves (all models)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=800, height=500)
    return fig

def confusion_matrix_fig(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Viridis', showscale=True, text=cm, texttemplate="%{text}"))
    fig.update_layout(title=title, width=520, height=420)
    return fig

def feature_importance_fig(model, feature_names, title="Feature importances"):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        inds = np.argsort(imp)[::-1]
        fig = go.Figure([go.Bar(x=[feature_names[i] for i in inds], y=imp[inds])])
        fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Importance", width=800, height=400)
        return fig
    else:
        return None

# --- App UI ---
menu = st.sidebar.selectbox("Select page", ["Overview (Insights)","Models (Train & Evaluate)","Predict (Upload & Score)","Data & Dictionary"])

df = load_sample_data()

if menu == "Overview (Insights)":
    st.header("Exploratory Insights — action-oriented charts for Marketing")
    st.markdown("Choose segments and see business-focused insights. (These charts assume the presence of the same Universal Bank fields.)")
    st.dataframe(df.head(5))

    # Chart 1: Acceptance rate by Education and Family (heatmap / pivot) — actionable: which education+family segments to target
    st.subheader("1) Acceptance rate by Education and Family (heatmap) — target the best segments")
    pivot = pd.pivot_table(df, values='Personal Loan', index='Education', columns='Family', aggfunc='mean')
    st.write("Acceptance rate (mean) for Education x Family")
    fig1 = px.imshow(pivot.values, x=pivot.columns.astype(str), y=["Edu_"+str(i) for i in pivot.index], text_auto=".3f", aspect="auto", labels=dict(x="Family size", y="Education Level"))
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Income distribution split by Personal Loan acceptance (overlaid histograms) - target income bands
    st.subheader("2) Income distribution by Loan acceptance — pick income bands for campaigns")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=df[df['Personal Loan']==0]['Income'], name='No', opacity=0.6, nbinsx=30))
    fig2.add_trace(go.Histogram(x=df[df['Personal Loan']==1]['Income'], name='Yes', opacity=0.6, nbinsx=30))
    fig2.update_layout(barmode='overlay', title='Income distribution: No vs Yes', xaxis_title='Income ($000)', yaxis_title='Count', width=900, height=450)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Binned Income vs Observed Acceptance Rate (business rule insight)
    st.subheader("3) Observed acceptance rate by Income bins (binned conversion rate) — use for thresholding")
    df['Income_bin'] = pd.cut(df['Income'], bins=[0,25,50,75,100,150,300], labels=['0-25','25-50','50-75','75-100','100-150','150+'])
    bin_rates = df.groupby('Income_bin')['Personal Loan'].mean().reset_index().rename(columns={'Personal Loan':'AcceptanceRate'})
    fig3 = px.bar(bin_rates, x='Income_bin', y='AcceptanceRate', text='AcceptanceRate', labels={'Income_bin':'Income bin', 'AcceptanceRate':'Acceptance rate'})
    fig3.update_layout(width=800, height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Feature importance from a quick RandomForest (proxy) — which levers move probability (actionable)
    st.subheader("4) Quick feature importance (RandomForest) — prioritize data-driven levers")
    try:
        X, y = preprocess(df)
        res, scaler, Xs, X_train, X_test, y_train, y_test = train_models(X, y)
        rf = res['Random Forest']['model']
        fig4 = feature_importance_fig(rf, res['Random Forest']['feature_names'], title="Random Forest - Feature importance")
        if fig4 is not None:
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.write("Model does not expose feature importances.")
    except Exception as e:
        st.write("Could not compute feature importances: ", e)

    # Chart 5: Age vs CCAvg scatter with acceptance rate color — find demographic product-market fit
    st.subheader("5) Age vs CCAvg scatter (color = Personal Loan) — demographic targeting")
    fig5 = px.scatter(df, x='Age', y='CCAvg', color=df['Personal Loan'].astype(str), size='Income', hover_data=['Income','Education','Family'], title='Age vs CCAvg (color=PersonalLoan)')
    fig5.update_layout(width=900, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("**Action ideas**:")
    st.write("- Target high-acceptance income bins with personalized offers.")
    st.write("- Prioritise customers with high CCAvg and higher education for focused campaigns.")
    st.write("- Use feature importance to decide which product-ownership signals (CD/Online/CreditCard) to bundle with offers.")

elif menu == "Models (Train & Evaluate)":
    st.header("Train models and view performance (Decision Tree, Random Forest, Gradient Boosting)")
    st.write("Upload a dataset or use sample data. Models will be trained on the dataset and evaluated (5-fold CV on training).")
    uploaded = st.file_uploader("Upload CSV with Universal Bank columns (or leave blank to use sample)", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("File loaded.")
    st.write("Dataset rows:", df.shape[0], "columns:", df.shape[1])
    st.dataframe(df.head(5))

    run = st.button("Train all 3 models and compute metrics (5-fold CV)")
    if run:
        try:
            X, y = preprocess(df)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()
        with st.spinner("Training models..."):
            results, scaler, Xs, X_train, X_test, y_train, y_test = train_models(X, y)
        st.success("Training complete. Metrics below:")

        # Metrics table
        mt = metrics_table(results)
        st.dataframe(mt)

        # ROC overlay
        st.subheader("ROC Curves (all models)")
        st.plotly_chart(plot_roc_all(results), use_container_width=True)

        # Confusion matrices (train + test)
        st.subheader("Confusion Matrices (training & testing)")
        cols = st.columns(3)
        for i,(name,v) in enumerate(results.items()):
            cols[i].markdown(f"**{name} — Test**")
            cols[i].plotly_chart(confusion_matrix_fig(v['y_test'], v['y_test_pred'], title=f"{name} - Test CM"))
        st.write("You can inspect training confusion matrices below:")
        for name,v in results.items():
            st.plotly_chart(confusion_matrix_fig(y_train, v['model'].predict(X_train), title=f"{name} - Train CM"))

        # Feature importances
        st.subheader("Feature importances (per model)")
        for name,v in results.items():
            fig_imp = feature_importance_fig(v['model'], v['feature_names'], title=f"{name} - Feature importances")
            if fig_imp is not None:
                st.plotly_chart(fig_imp)
            else:
                st.write(f"{name} does not provide feature_importances_")

        # Save models into session state for prediction tab
        st.session_state['trained_models'] = {k:v['model'] for k,v in results.items()}
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = results[next(iter(results))]['feature_names']

elif menu == "Predict (Upload & Score)":
    st.header("Upload new dataset and predict 'Personal Loan' label")
    st.markdown("Upload a CSV with the same columns (including target if you want to compare). The app will predict label and provide a download link of predictions.")
    uploaded = st.file_uploader("Upload CSV to predict", type=['csv'])
    if uploaded is not None:
        newdf = pd.read_csv(uploaded)
        st.write("Uploaded:", newdf.shape)
        st.dataframe(newdf.head(5))
        # Ensure models trained or train on sample
        if 'trained_models' not in st.session_state:
            st.warning("No trained models in session — training on available dataset automatically to create models for prediction.")
            try:
                X, y = preprocess(df)
                results, scaler, Xs, X_train, X_test, y_train, y_test = train_models(X, y)
                st.session_state['trained_models'] = {k:v['model'] for k,v in results.items()}
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = results[next(iter(results))]['feature_names']
            except Exception as e:
                st.error("Could not train models automatically: " + str(e))
                st.stop()
        models = st.session_state['trained_models']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']

        # Preprocess new data (drop ID/zip)
        for c in ['ID','Zip Code','ZIP Code','Zip','Zipcode']:
            if c in newdf.columns:
                newdf = newdf.drop(columns=[c])
        # Keep only features model expects (if present)
        X_new = newdf.copy()
        missing = [f for f in feature_names if f not in X_new.columns]
        if missing:
            st.warning(f"Uploaded data is missing features expected by model: {missing}. Predictions will fail unless you provide these columns.")
        # Fill missing numeric with 0 to allow predictions (quick fallback)
        X_new = X_new.reindex(columns=feature_names, fill_value=0)
        # scale continuous columns if scaler exists
        cont = [c for c in feature_names if c.lower() in ['age','experience','income','ccavg','mortgage']]
        if cont and scaler is not None:
            X_new[cont] = scaler.transform(X_new[cont])
        # Predict with chosen model
        model_choice = st.selectbox("Choose model for prediction", list(models.keys()))
        model = models[model_choice]
        preds = model.predict(X_new)
        proba = model.predict_proba(X_new)[:,1] if hasattr(model,'predict_proba') else np.zeros(len(preds))
        newdf['Predicted_PersonalLoan'] = preds.astype(int)
        newdf['Pred_Prob'] = np.round(proba,4)
        st.success("Predictions added to the table below.")
        st.dataframe(newdf.head(10))

        # Download predicted CSV
        towrite = BytesIO()
        newdf.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download predictions CSV", data=towrite, file_name="predictions.csv", mime="text/csv")

elif menu == "Data & Dictionary":
    st.header("Data sample and Universal Bank data dictionary")
    st.dataframe(df.head(10))
    st.markdown("**Universal Bank Data Fields**")
    dd = pd.DataFrame({
        'Field':['ID','Personal Loan','Age','Experience','Income','Zip code','Family','CCAvg','Education','Mortgage','Securities','CDAccount','Online','CreditCard'],
        'Description':[
            'unique identifier',
            'did the customer accept the personal loan offered (1=Yes, 0=No)',
            'customer’s age',
            'number of years of profession experience',
            'annual income of the customer ($000)',
            'home address zip code',
            'family size of customer',
            'average spending on credit cards per month ($000)',
            'education level (1) undergraduate, (2) graduate, (3) advanced/professional',
            'value of house mortgage ($000)',
            'does the customer have a securities account with the bank? (1=Yes, 0=No)',
            'does the customer have a certificate of deposit with the bank? (1=Yes, 0=No)',
            'does the customer use Internet banking facilities (1=Yes, 0=No)',
            'does the customer use a credit card issued by Universal Bank? (1=Yes, 0=No)'
        ]
    })
    st.dataframe(dd)

    st.markdown("**Notes**: This app trains models on the dataset you upload. For production use in Streamlit Cloud you may want to pretrain models and load them, or implement scheduled retraining.")
