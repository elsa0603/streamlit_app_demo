# streamlit_app.py
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# UMAP 不是 sklearn 內建，requirements 已包含 umap-learn
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

st.set_page_config(page_title="Iris 分類可視化 App（Matplotlib 版）", layout="wide")

# -----------------------------
# 載入資料
# -----------------------------
@st.cache_data
def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names
    return X, y, target_names

X, y, target_names = load_iris()

# Feature 別名（中文）
feature_alias = {
    "sepal length (cm)": "花萼長",
    "sepal width (cm)": "花萼寬",
    "petal length (cm)": "花瓣長",
    "petal width (cm)": "花瓣寬",
}

# -----------------------------
# 側邊欄：模型與降維方式
# -----------------------------
st.sidebar.title("設定")
model_name = st.sidebar.selectbox(
    "選擇分類模型",
    ["Logistic Regression", "SVM (RBF)", "Random Forest"],
    index=0,  # 明確指定 index，避免狀態還原產生越界
    key="model_name_v1"
)

# 基本參數
if model_name == "Logistic Regression":
    C = st.sidebar.slider("C（正則化強度倒數）", 0.01, 10.0, 1.0, 0.01, key="lr_C")
elif model_name == "SVM (RBF)":
    C = st.sidebar.slider("C（懲罰係數）", 0.1, 10.0, 1.0, 0.1, key="svm_C")
    gamma = st.sidebar.select_slider("gamma", options=["scale", "auto"], value="scale", key="svm_gamma")
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("樹的數量 n_estimators", 50, 400, 200, 10, key="rf_n")
    max_depth = st.sidebar.slider("最大深度 max_depth (0 為 None)", 0, 20, 0, 1, key="rf_depth")

dr_options = ["PCA", "t-SNE"]
if HAS_UMAP:
    dr_options.append("UMAP")

# 明確指定安全預設 index，並使用依 HAS_UMAP 變動的 key 以避免舊的 index 被沿用
default_dr = "PCA" if "PCA" in dr_options else dr_options[0]
dr_method = st.sidebar.selectbox(
    "降維方法（2D）",
    dr_options,
    index=dr_options.index(default_dr),
    key=f"dr_method_{int(HAS_UMAP)}",
)

random_state = st.sidebar.number_input("random_state（穩定重現）", min_value=0, value=42, step=1, key="rand_state")

# -----------------------------
# 建立模型
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_model(model_name, params, random_state):
    if model_name == "Logistic Regression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=params["C"], max_iter=1000, multi_class="auto"))
        ])
    elif model_name == "SVM (RBF)":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=params["C"], gamma=params["gamma"], probability=True, random_state=random_state))
        ])
    else:  # Random Forest
        md = params["max_depth"] if params["max_depth"] > 0 else None
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=md, random_state=random_state)
    return model

params = {}
if model_name == "Logistic Regression":
    params["C"] = C
elif model_name == "SVM (RBF)":
    params["C"] = C
    params["gamma"] = gamma
else:
    params["n_estimators"] = n_estimators
    params["max_depth"] = max_depth

model = build_model(model_name, params, random_state)

# 交叉驗證（顯示概略表現）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
st.sidebar.markdown(f"**CV (5-fold) 準確率**：{scores.mean():.3f} ± {scores.std():.3f}")

# -----------------------------
# 使用者輸入（四個特徵）
# -----------------------------
def number_slider(label, series, step=0.1, key=None):
    return st.slider(
        label,
        float(series.min()),
        float(series.max()),
        float(series.mean()),
        step=step,
        key=key,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("輸入要預測的樣本")
sl = number_slider(f"{feature_alias['sepal length (cm)']}（cm）", X["sepal length (cm)"], key="sl")
sw = number_slider(f"{feature_alias['sepal width (cm)']}（cm）", X["sepal width (cm)"], key="sw")
pl = number_slider(f"{feature_alias['petal length (cm)']}（cm）", X["petal length (cm)"], key="pl")
pw = number_slider(f"{feature_alias['petal width (cm)']}（cm）", X["petal width (cm)"], key="pw")

user_X = np.array([[sl, sw, pl, pw]])

# -----------------------------
# 訓練模型並預測
# -----------------------------
model.fit(X, y)
pred_class_idx = model.predict(user_X)[0]
try:
    proba = model.predict_proba(user_X)[0]
except Exception:
    proba = np.array([np.nan] * len(target_names))

pred_class = target_names[pred_class_idx]

# -----------------------------
# 降維並準備 2D 資料
# -----------------------------
@st.cache_data(show_spinner=False)
def reduce_dim_and_project(X_df, method, user_X, random_state=42):
    subtitle = ""
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        components = reducer.fit_transform(X_df)
        explained = reducer.explained_variance_ratio_
        subtitle = f"PCA（解釋變異 {explained[0]:.2%} + {explained[1]:.2%}）"
        proj_user = reducer.transform(user_X)[0]
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        components = reducer.fit_transform(X_df)
        subtitle = "t-SNE"
        # 用最近鄰的 2D 位置近似
        dists = np.linalg.norm(X_df.values - user_X, axis=1)
        proj_user = components[np.argmin(dists)]
    elif method == "UMAP" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        components = reducer.fit_transform(X_df)
        subtitle = "UMAP"
        dists = np.linalg.norm(X_df.values - user_X, axis=1)
        proj_user = components[np.argmin(dists)]
    else:
        reducer = PCA(n_components=2, random_state=random_state)
        components = reducer.fit_transform(X_df)
        explained = reducer.explained_variance_ratio_
        subtitle = f"PCA（解釋變異 {explained[0]:.2%} + {explained[1]:.2%}）"
        proj_user = reducer.transform(user_X)[0]

    df2d = pd.DataFrame(components, columns=["dim1", "dim2"])
    return df2d, subtitle, proj_user

df2d, subtitle, user_point = reduce_dim_and_project(X, dr_method, user_X, random_state=random_state)
df2d["target"] = y.map(lambda i: target_names[i])

# -----------------------------
# Matplotlib 繪圖（不指定色彩與樣式）
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots()
    # 逐類別畫散點（顏色由 Matplotlib 預設循環）
    for cls in target_names:
        data = df2d[df2d["target"] == cls]
        ax.scatter(data["dim1"], data["dim2"], label=cls)
    # 使用者點（以 'x' 標記）
    ax.scatter([user_point[0]], [user_point[1]], marker="x", s=100, label="Your sample")
    ax.set_title(f"Iris 2D 視覺化（{subtitle}）")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("預測結果")
    st.markdown(f"**模型**：{model_name}")
    st.markdown(f"**預測品種**：:blue[{pred_class}]")
    if not np.isnan(proba).any():
        proba_df = pd.DataFrame({"class": target_names, "probability": proba}).sort_values("probability", ascending=False)
        st.dataframe(proba_df, use_container_width=True)
    else:
        st.info("此模型目前無機率輸出（predict_proba），僅顯示類別預測。")
    st.markdown("---")
    st.caption("💡 左側可調整降維方法與模型參數，並拖動滑桿輸入特徵值。")

st.markdown("---")
with st.expander("顯示資料預覽與描述統計"):
    st.dataframe(pd.concat([X, y.rename('target_idx')], axis=1).head())
    st.write("描述統計：")
    st.write(X.describe())

st.caption("Built with Streamlit · scikit-learn · Matplotlib")
