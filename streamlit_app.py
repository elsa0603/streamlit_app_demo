# streamlit_knn_classifier.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="KNN 分類（可調 K 值）", layout="wide")

# -----------------------------
# 資料載入
# -----------------------------
@st.cache_data
def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names
    return X, y, target_names

X, y, target_names = load_iris()

# -----------------------------
# 側邊欄設定
# -----------------------------
st.sidebar.header("設定")
# 使用兩個特徵便於 2D 視覺化
feat1 = st.sidebar.selectbox("特徵 1", list(X.columns), index=2)
feat2 = st.sidebar.selectbox("特徵 2", [c for c in X.columns if c != feat1], index=3)

k = st.sidebar.slider("K 值（近鄰數）", min_value=1, max_value=30, value=5, step=1)
weights = st.sidebar.selectbox("權重(weights)", ["uniform", "distance"], index=0)
metric = st.sidebar.selectbox("距離度量(metric)", ["minkowski", "euclidean", "manhattan"], index=0)
test_size = st.sidebar.slider("測試集比例", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

st.sidebar.caption("不使用交叉驗證；直接 train/test split 後訓練並回報測試集精確度。")

# -----------------------------
# 準備資料（僅兩個特徵）
# -----------------------------
X2 = X[[feat1, feat2]].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=test_size, random_state=int(random_state), stratify=y
)

# -----------------------------
# 建立與訓練模型
# -----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric))
])

pipe.fit(X_train, y_train)

# -----------------------------
# 評估（直接在測試集）
# -----------------------------
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# 視覺化（決策邊界 + 訓練/測試散點）
# -----------------------------
def plot_decision_boundary(pipe, X_train, y_train, X_test, y_test, feat1, feat2, target_names):
    # 建網格
    x_min = min(X_train[feat1].min(), X_test[feat1].min()) - 0.5
    x_max = max(X_train[feat1].max(), X_test[feat1].max()) + 0.5
    y_min = min(X_train[feat2].min(), X_test[feat2].min()) - 0.5
    y_max = max(X_train[feat2].max(), X_test[feat2].max()) + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pipe.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    # 決策區域
    ax.contourf(xx, yy, Z, alpha=0.2)

    # 訓練集點
    for i, name in enumerate(target_names):
        m = (y_train.values == i)
        ax.scatter(X_train.loc[m, feat1], X_train.loc[m, feat2], label=f"train-{name}", marker="o", s=25)

    # 測試集點（以邊框區分）
    for i, name in enumerate(target_names):
        m = (y_test.values == i)
        ax.scatter(X_test.loc[m, feat1], X_test.loc[m, feat2], label=f"test-{name}", marker="^", s=35, edgecolor="k", linewidths=0.5)

    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title("KNN 決策邊界與資料散點")
    ax.legend(loc="best", fontsize=8)
    return fig

# -----------------------------
# 版面配置
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    fig = plot_decision_boundary(pipe, X_train, y_train, X_test, y_test, feat1, feat2, target_names)
    st.pyplot(fig)

with col2:
    st.subheader("測試集績效")
    st.markdown(f"**Accuracy**：`{acc:.3f}`")
    st.markdown("---")
    st.write("混淆矩陣（test）：")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names), use_container_width=True)
    st.markdown("---")
    st.write("分類報告（test）：")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

st.markdown("---")
with st.expander("資料預覽"):
    st.dataframe(pd.concat([X, y.rename("target_idx")], axis=1).head())

st.caption("Built with Streamlit · scikit-learn · Matplotlib · Iris dataset")
