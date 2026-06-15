import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel


# ============================================================
# Page setup
# ============================================================

st.set_page_config(page_title="Quantum Kernel SIR App", layout="wide")

st.title("Quantum Kernel SIR: Qiskit 기반 차원축소 앱")

st.markdown(
    r"""
    이 앱은 통계학의 **Sliced Inverse Regression(SIR)** 과 양자 머신러닝의
    **Quantum Kernel**을 결합한 실험용 앱입니다.

    핵심 질문은 다음입니다.

    > 데이터 \(X\)에서 response \(Y\)와 관련 있는 저차원 표현을 찾을 때,
    > 기존 classical kernel 대신 Qiskit quantum feature map으로 만든 quantum kernel을
    > 사용하면 어떤 결과가 나오는가?

    이 프로젝트는 완전한 양자 알고리즘으로 SIR 전체를 구현한 것이 아니라,
    **Kernel SIR의 kernel evaluation step을 quantum kernel로 대체한
    hybrid quantum-classical 구현**입니다.
    """
)


# ============================================================
# Explanation functions
# ============================================================

def render_project_overview():
    st.header("1. 프로젝트 개요")
    st.markdown(
        r"""
        이 앱은 데이터 \(X\)와 response \(Y\)가 주어졌을 때,
        \(Y\)를 설명하는 데 중요한 저차원 표현을 찾는 세 가지 방법을 비교합니다.

        | 방법 | 설명 |
        |---|---|
        | **Linear SIR** | 원래 데이터 공간에서 \(Y\)-관련 선형 방향을 찾는 방법 |
        | **RBF Kernel SIR** | RBF kernel이 유도하는 비선형 feature space에서 SIR을 수행하는 방법 |
        | **Quantum Kernel SIR** | Qiskit quantum feature map으로 계산한 quantum kernel을 사용해 SIR을 수행하는 방법 |

        전체 계산 흐름은 다음과 같습니다.
        """
    )
    st.latex(
        r"""
        X,Y
        \longrightarrow
        \text{Kernel Matrix}
        \longrightarrow
        \text{Kernel SIR}
        \longrightarrow
        \text{2D Embedding}
        \longrightarrow
        \text{Accuracy or }R^2
        """
    )
    st.info(
        """
        핵심은 “SIR 전체를 양자회로로 구현했다”가 아니라,
        기존 Kernel SIR에서 사용하던 classical kernel matrix를
        Qiskit quantum kernel matrix로 대체했다는 점입니다.
        """
    )


def render_sdr_sir_explanation():
    with st.expander("2. 배경 설명: SDR과 SIR", expanded=True):
        st.markdown(
            r"""
            ## 2.1 Sufficient Dimension Reduction, SDR

            통계학에서 자주 다루는 문제는 covariate vector \(X\)로 response \(Y\)를
            설명하는 것입니다. 그런데 \(X\)가 고차원이면, \(X\) 전체를 그대로 쓰는 대신
            \(Y\)를 설명하는 데 필요한 저차원 summary를 찾고 싶습니다.

            SDR의 목표는 다음 조건을 만족하는 projection을 찾는 것입니다.
            """
        )
        st.latex(r"""Y \perp X \mid B^T X""")
        st.markdown(
            r"""
            이 조건의 의미는 다음과 같습니다.

            > \(B^T X\)를 알고 나면, \(X\) 전체를 더 알아도 \(Y\)에 대한 추가 정보가 없다.

            즉 \(B^T X\)는 \(Y\)를 설명하는 데 충분한 저차원 표현입니다.

            ## 2.2 Sliced Inverse Regression, SIR

            SIR은 SDR 방법 중 하나입니다. 보통 회귀에서는 \(Y\mid X\)를 직접 보지만,
            SIR은 반대로 \(X\mid Y\)를 봅니다.

            핵심 아이디어는 다음입니다.

            > \(Y\)가 달라질 때 \(X\)의 평균이 어느 방향으로 움직이는지를 보면,
            > \(Y\)와 관련 있는 \(X\)-방향을 찾을 수 있다.

            이를 위해 \(Y\)를 여러 그룹, 즉 slice로 나눕니다. classification 문제에서는
            class label 자체가 slice가 됩니다.

            각 slice \(h\)에서 \(X\)의 평균을 계산합니다.
            """
        )
        st.latex(
            r"""
            \bar{x}_h
            =
            \frac{1}{n_h}
            \sum_{i:Y_i\in h} x_i
            """
        )
        st.markdown("그다음 다음과 같은 SIR matrix를 구성합니다.")
        st.latex(
            r"""
            \widehat M_{\mathrm{SIR}}
            =
            \sum_h \hat p_h \bar{x}_h\bar{x}_h^T
            """
        )
        st.markdown(
            r"""
            이 행렬의 큰 eigenvalue에 대응하는 eigenvector들이 response \(Y\)와 관련 있는
            중요한 방향으로 해석됩니다.
            """
        )


def render_kernel_sir_explanation():
    with st.expander("3. Kernel SIR: 비선형 구조를 잡기 위한 확장", expanded=True):
        st.markdown(
            r"""
            ## 3.1 Linear SIR의 한계

            Linear SIR은 다음과 같은 선형 projection만 찾습니다.
            """
        )
        st.latex(r"""X \mapsto B^T X""")
        st.markdown(
            r"""
            하지만 실제 데이터에서는 \(Y\)가 \(X\)의 선형결합이 아니라 비선형 함수에
            의존할 수 있습니다. 예를 들어 circles 데이터는 대략 다음과 같은 구조를 가집니다.
            """
        )
        st.latex(r"""Y = 1\{X_1^2 + X_2^2 > c\}""")
        st.markdown(
            r"""
            이 경우 중요한 정보는 \(X_1\) 또는 \(X_2\)의 선형 방향이 아니라
            \(X_1^2+X_2^2\) 같은 비선형 feature입니다.

            ## 3.2 Kernel trick

            Kernel method는 원래 데이터 \(x\)를 더 큰 feature space로 보낸다고 생각합니다.
            """
        )
        st.latex(r"""x \mapsto \Phi(x)\in\mathcal H""")
        st.markdown(
            r"""
            여기서 \(\mathcal H\)는 Hilbert space 또는 RKHS입니다.
            직접 \(\Phi(x)\)를 계산하지 않고, inner product만 kernel function으로 계산합니다.
            """
        )
        st.latex(
            r"""
            K(x_i,x_j)
            =
            \langle\Phi(x_i),\Phi(x_j)\rangle_{\mathcal H}
            """
        )
        st.markdown("대표적인 classical kernel은 RBF kernel입니다.")
        st.latex(
            r"""
            K_{\mathrm{RBF}}(x,z)
            =
            \exp(-\gamma\|x-z\|^2)
            """
        )
        st.markdown(
            r"""
            RBF kernel은 가까운 점끼리는 similarity를 크게, 먼 점끼리는 similarity를 작게
            부여합니다.

            ## 3.3 Kernel SIR

            Kernel SIR은 원래 \(X\)-space가 아니라 feature space \(\mathcal H\)에서 SIR을
            수행합니다. slice \(h\)에서 feature-space 평균은 다음과 같습니다.
            """
        )
        st.latex(
            r"""
            \hat\mu_h
            =
            \frac{1}{n_h}
            \sum_{i:Y_i\in h}\Phi(x_i)
            """
        )
        st.markdown("Kernel SIR operator는 다음처럼 생각할 수 있습니다.")
        st.latex(
            r"""
            \widehat M_{\mathrm{KSIR}}
            =
            \sum_h \hat p_h\, \hat\mu_h\otimes\hat\mu_h
            """
        )
        st.markdown(
            r"""
            이 앱에서는 \(\Phi(x)\)를 직접 계산하지 않고, kernel matrix만으로 feature-space에서의
            SIR 계산을 수행합니다.
            """
        )


def render_quantum_kernel_explanation():
    with st.expander("4. Quantum Kernel은 어디에 들어가는가?", expanded=True):
        st.markdown(
            r"""
            ## 4.1 Quantum feature map

            이 프로젝트의 양자 부분은 **quantum feature map**입니다.
            classical data point \(x\)를 quantum circuit에 넣어 quantum state로 encoding합니다.
            """
        )
        st.latex(
            r"""
            |\phi(x)\rangle
            =
            U_\phi(x)|0\rangle^{\otimes q}
            """
        )
        st.markdown(
            r"""
            여기서

            - \(q\): qubit 수
            - \(U_\phi(x)\): 데이터 \(x\)에 따라 달라지는 quantum circuit
            - \(|0\rangle^{\otimes q}\): 모든 qubit이 0인 초기 상태

            이 앱에서는 Qiskit의 `ZZFeatureMap`을 사용합니다.

            ## 4.2 Quantum kernel

            두 데이터 \(x_i,x_j\)의 similarity를 두 quantum state의 overlap으로 정의합니다.
            """
        )
        st.latex(
            r"""
            K_Q(x_i,x_j)
            =
            |\langle\phi(x_i)\mid\phi(x_j)\rangle|^2
            """
        )
        st.markdown("회로 관점에서는 다음처럼 쓸 수 있습니다.")
        st.latex(
            r"""
            K_Q(x_i,x_j)
            =
            \left|
            \langle 0|U_\phi(x_i)^\dagger U_\phi(x_j)|0\rangle
            \right|^2
            """
        )
        st.markdown(
            r"""
            즉 다음 과정을 통해 kernel value를 해석할 수 있습니다.

            1. \(|0\cdots0\rangle\)에서 시작한다.
            2. \(U_\phi(x_j)\)를 적용한다.
            3. \(U_\phi(x_i)^\dagger\)를 적용한다.
            4. 다시 \(|0\cdots0\rangle\)로 돌아올 확률을 계산한다.

            이 확률이 quantum kernel entry입니다.

            ## 4.3 양자 알고리즘 접목의 의미

            이 프로젝트는 SIR 전체를 quantum computer에서 수행한 것은 아닙니다.
            대신 Kernel SIR에 필요한 kernel matrix를 quantum feature map으로부터 계산합니다.
            즉 다음 대체를 수행합니다.
            """
        )
        st.latex(r"""K_{\mathrm{RBF}}\quad\longrightarrow\quad K_Q""")
        st.markdown("따라서 전체 구조는 hybrid quantum-classical pipeline입니다.")
        st.latex(
            r"""
            \text{Classical Data}
            \longrightarrow
            \text{Quantum Feature Map}
            \longrightarrow
            \text{Quantum Kernel Matrix}
            \longrightarrow
            \text{Classical Kernel SIR}
            """
        )


def render_how_to_read_results():
    with st.expander("5. 결과 화면을 읽는 방법", expanded=True):
        st.markdown(
            r"""
            앱을 실행하면 다음 결과들이 표시됩니다.

            | 출력 | 의미 |
            |---|---|
            | Original Data | 원래 데이터의 첫 두 feature를 시각화한 그림 |
            | RBF Kernel Matrix | RBF kernel로 계산한 sample 간 similarity |
            | Quantum Kernel Matrix | quantum feature map으로 계산한 sample 간 similarity |
            | Linear SIR Embedding | 선형 SIR이 찾은 2차원 표현 |
            | RBF Kernel SIR Embedding | RBF kernel 기반 Kernel SIR 표현 |
            | Quantum Kernel SIR Embedding | quantum kernel 기반 Kernel SIR 표현 |
            | Accuracy or \(R^2\) | embedding이 \(Y\)-관련 정보를 얼마나 보존하는지 보는 지표 |
            | Quantum Circuit | 데이터 encoding에 사용된 Qiskit feature map 회로 |

            ## Classification일 때

            각 embedding 위에서 logistic regression을 학습하고 accuracy를 계산합니다.
            """
        )
        st.latex(
            r"""
            \mathrm{Accuracy}
            =
            \frac{\#\{\hat y_i = y_i\}}{n_{\mathrm{test}}}
            """
        )
        st.markdown(
            r"""
            Accuracy가 높다는 것은 해당 2D embedding이 class 정보를 잘 보존한다는 뜻입니다.

            ## Regression일 때

            각 embedding 위에서 linear regression을 학습하고 \(R^2\)를 계산합니다.
            """
        )
        st.latex(
            r"""
            R^2
            =
            1-
            \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
            """
        )
        st.markdown(
            r"""
            \(R^2\)가 높다는 것은 해당 embedding이 continuous response \(Y\)를 잘 설명한다는 뜻입니다.

            중요한 점은 Quantum Kernel SIR이 항상 가장 좋아야 하는 것은 아니라는 점입니다.
            이 앱의 목적은 quantum advantage를 주장하는 것이 아니라, quantum kernel이
            Kernel SIR pipeline에 어떻게 접목될 수 있는지 보여주는 것입니다.
            """
        )


def render_limitations():
    with st.expander("6. 한계와 주의점", expanded=False):
        st.markdown(
            r"""
            이 프로젝트의 한계는 다음과 같습니다.

            1. **실제 양자 하드웨어가 아니라 simulator를 사용했습니다.**  
               이 앱은 Qiskit statevector 기반 quantum kernel 계산을 사용합니다.

            2. **Quantum advantage를 증명한 것이 아닙니다.**  
               sample size가 작고 toy dataset 중심이므로 결과는 exploratory하게 해석해야 합니다.

            3. **성능은 dataset geometry에 민감합니다.**  
               예를 들어 moons 데이터에서는 RBF kernel이 더 자연스럽게 잘 맞을 수 있습니다.

            4. **Quantum feature map parameter에 민감합니다.**  
               `ZZFeatureMap reps`를 키운다고 항상 좋아지는 것은 아닙니다.

            5. **Kernel SIR 구현은 finite-sample empirical version입니다.**  
               theoretical central subspace를 완벽히 복원했다고 주장하는 것이 아니라,
               sample 기반으로 slice mean 구조를 계산한 것입니다.
            """
        )
        st.latex(
            r"""
            \text{Quantum kernel can be inserted into Kernel SIR,}
            \quad
            \text{but its performance is data- and parameter-dependent.}
            """
        )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Settings")

show_explanation = st.sidebar.checkbox("Show detailed explanation", value=True)

dataset_name = st.sidebar.selectbox(
    "Dataset",
    [
        "Synthetic nonlinear regression",
        "Moons classification",
        "Circles classification",
        "Iris 3-class classification",
    ],
)

n_samples = st.sidebar.slider("Number of samples", 20, 80, 40, step=10)
noise = st.sidebar.slider("Noise", 0.01, 0.40, 0.10, step=0.01)
n_slices = st.sidebar.slider("Number of slices for regression", 3, 8, 5, step=1)
rbf_gamma = st.sidebar.slider("RBF gamma", 0.1, 5.0, 1.0, step=0.1)
q_reps = st.sidebar.slider("ZZFeatureMap reps", 1, 3, 1, step=1)
test_size = st.sidebar.slider("Test size", 0.2, 0.5, 0.3, step=0.05)

run_button = st.button("Run Kernel SIR Experiment")


# ============================================================
# Show explanations
# ============================================================

if show_explanation:
    render_project_overview()
    render_sdr_sir_explanation()
    render_kernel_sir_explanation()
    render_quantum_kernel_explanation()
    render_how_to_read_results()
    render_limitations()


# ============================================================
# Dataset generation
# ============================================================

def load_dataset(name, n_samples, noise):
    """
    Returns X, y, task_type.
    task_type is either 'regression' or 'classification'.
    """
    rng = np.random.default_rng(42)

    if name == "Synthetic nonlinear regression":
        n = n_samples
        p = 4
        X = rng.normal(size=(n, p))
        y = (
            np.sin(2.0 * X[:, 0])
            + 0.8 * X[:, 1] ** 2
            - 0.5 * X[:, 2]
            + noise * rng.normal(size=n)
        )
        return X, y, "regression"

    if name == "Moons classification":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        return X, y, "classification"

    if name == "Circles classification":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        return X, y, "classification"

    # Iris 3-class classification
    iris = load_iris()
    X_all = iris.data[:, :4]
    y_all = iris.target

    labels = np.unique(y_all)
    per_class = max(2, n_samples // len(labels))

    chosen_idx = []
    for label in labels:
        idx_label = np.where(y_all == label)[0]
        k = min(per_class, len(idx_label))
        chosen_idx.extend(rng.choice(idx_label, size=k, replace=False))

    chosen_idx = np.array(chosen_idx)
    if len(chosen_idx) > n_samples:
        chosen_idx = rng.choice(chosen_idx, size=n_samples, replace=False)

    X = X_all[chosen_idx]
    y = y_all[chosen_idx]
    return X, y, "classification"


# ============================================================
# Slicing and SIR functions
# ============================================================

def make_slices(y, task_type, n_slices):
    """
    For classification, slices are class labels.
    For regression, slices are quantile bins of y.
    """
    y = np.asarray(y)

    if task_type == "classification":
        return y.astype(int)

    quantiles = np.linspace(0, 1, n_slices + 1)
    cut_points = np.quantile(y, quantiles)
    cut_points = np.unique(cut_points)

    if len(cut_points) <= 2:
        ranks = np.argsort(np.argsort(y))
        slices = np.floor(n_slices * ranks / len(y)).astype(int)
        return np.minimum(slices, n_slices - 1)

    return np.digitize(y, cut_points[1:-1], right=True).astype(int)


def center_kernel(K):
    """Double-center a kernel matrix: K_c = H K H."""
    K = np.asarray(K)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_sir_embedding(X, slices, n_components=2):
    """Classical empirical SIR."""
    X = np.asarray(X)
    slices = np.asarray(slices)
    _, p = X.shape

    M = np.zeros((p, p))
    for h in np.unique(slices):
        idx = slices == h
        if np.sum(idx) == 0:
            continue
        p_h = np.mean(idx)
        mean_h = X[idx].mean(axis=0).reshape(-1, 1)
        M += p_h * (mean_h @ mean_h.T)

    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    d = min(n_components, p)
    B = eigvecs[:, :d]
    Z = X @ B

    if Z.shape[1] == 1:
        Z = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])

    return Z[:, :2], eigvals


def kernel_sir_embedding(K, slices, n_components=2, ridge=1e-8):
    """
    Empirical Kernel SIR using slice means in RKHS.
    The computation is performed using only the centered kernel matrix.
    """
    K = np.asarray(K)
    slices = np.asarray(slices)
    n = K.shape[0]
    Kc = center_kernel(K)

    labels = np.unique(slices)
    B = np.zeros((n, len(labels)))

    for j, h in enumerate(labels):
        idx = slices == h
        n_h = np.sum(idx)
        if n_h == 0:
            continue
        p_h = n_h / n
        B[idx, j] = np.sqrt(p_h) / n_h

    G = B.T @ Kc @ B
    G = G + ridge * np.eye(G.shape[0])

    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    d = min(n_components, len(labels))
    Beta = eigvecs[:, :d]
    Z = Kc @ B @ Beta
    Z = StandardScaler().fit_transform(Z)

    if Z.shape[1] == 1:
        Z = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])

    return Z[:, :2], eigvals


# ============================================================
# Evaluation and plots
# ============================================================

def evaluate_embedding(Z, y, task_type, test_size):
    """For classification: logistic regression accuracy. For regression: linear regression R^2."""
    stratify = y if task_type == "classification" else None
    Z_train, Z_test, y_train, y_test = train_test_split(
        Z,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    if task_type == "classification":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z_train, y_train)
        pred = clf.predict(Z_test)
        return accuracy_score(y_test, pred)

    reg = LinearRegression()
    reg.fit(Z_train, y_train)
    pred = reg.predict(Z_test)
    return r2_score(y_test, pred)


def scatter_plot(Z, color, title, task_type):
    fig, ax = plt.subplots()
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if task_type == "regression":
        fig.colorbar(sc, ax=ax, label="y")
    return fig


def heatmap_plot(K, title):
    fig, ax = plt.subplots()
    im = ax.imshow(K)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig


# ============================================================
# Initial screen
# ============================================================

if not run_button:
    st.success(
        """
        왼쪽 sidebar에서 dataset과 parameter를 선택한 뒤
        **Run Kernel SIR Experiment** 버튼을 누르면 실험이 실행됩니다.
        """
    )
    st.stop()


# ============================================================
# Main execution
# ============================================================

st.header("7. 실험 결과")

X, y, task_type = load_dataset(dataset_name, n_samples, noise)
X_std = StandardScaler().fit_transform(X)
X_q = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_std)
slices = make_slices(y, task_type, n_slices)

st.subheader("7.1 Dataset Summary")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Dataset", dataset_name)
with col_b:
    st.metric("Task type", task_type)
with col_c:
    st.metric("Sample size", len(y))
with col_d:
    st.metric("Number of slices", len(np.unique(slices)))

st.markdown(
    r"""
    여기서 slice는 SIR이 \(Y\)-정보를 사용하는 방식입니다.  
    classification에서는 class label이 slice이고, regression에서는 \(Y\)를 quantile bin으로 나눕니다.
    """
)

st.subheader("7.2 Kernel Computation")
with st.spinner("Computing RBF kernel..."):
    K_rbf = rbf_kernel(X_std, X_std, gamma=rbf_gamma)

feature_dim = X_q.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=q_reps)
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

with st.spinner("Computing quantum kernel matrix. This may take a while..."):
    K_q = qkernel.evaluate(x_vec=X_q)

st.markdown(
    r"""
    아래 heatmap은 sample 간 similarity matrix를 보여줍니다.  
    밝은 색은 두 sample이 비슷하다는 뜻이고, 어두운 색은 덜 비슷하다는 뜻입니다.
    """
)

st.subheader("7.3 SIR Embeddings")
with st.spinner("Computing Linear SIR, RBF Kernel SIR, and Quantum Kernel SIR..."):
    Z_linear, eig_linear = linear_sir_embedding(X_std, slices, n_components=2)
    Z_rbf, eig_rbf = kernel_sir_embedding(K_rbf, slices, n_components=2)
    Z_q, eig_q = kernel_sir_embedding(K_q, slices, n_components=2)

score_linear = evaluate_embedding(Z_linear, y, task_type, test_size)
score_rbf = evaluate_embedding(Z_rbf, y, task_type, test_size)
score_q = evaluate_embedding(Z_q, y, task_type, test_size)

st.subheader("7.4 Original Data and Kernel Matrices")
c1, c2, c3 = st.columns(3)
with c1:
    st.pyplot(scatter_plot(X_std[:, :2], y, "Original data: first two standardized features", task_type))
    st.caption("원래 데이터의 첫 두 feature만 그린 그림입니다. 고차원 데이터일 경우 전체 구조를 완전히 보여주지는 않습니다.")
with c2:
    st.pyplot(heatmap_plot(K_rbf, "RBF Kernel Matrix"))
    st.caption("RBF kernel로 계산한 sample 간 similarity입니다. 가까운 점일수록 similarity가 높습니다.")
with c3:
    st.pyplot(heatmap_plot(K_q, "Quantum Kernel Matrix"))
    st.caption("Qiskit quantum feature map으로 만든 quantum state들의 fidelity를 kernel로 사용한 matrix입니다.")

metric_name = "Accuracy" if task_type == "classification" else "R^2"

st.subheader("7.5 Embedding Comparison")
st.markdown(
    r"""
    아래 그림들은 각 방법이 데이터를 2차원으로 줄인 결과입니다.  
    같은 색 또는 비슷한 response 값을 가진 점들이 잘 모이면, 해당 embedding이 \(Y\)-정보를 잘 보존한다고 볼 수 있습니다.
    """
)

d1, d2, d3 = st.columns(3)
with d1:
    st.pyplot(scatter_plot(Z_linear, y, f"Linear SIR, {metric_name}={score_linear:.3f}", task_type))
    st.caption("원래 feature space에서 선형 방향을 찾은 결과입니다.")
with d2:
    st.pyplot(scatter_plot(Z_rbf, y, f"RBF Kernel SIR, {metric_name}={score_rbf:.3f}", task_type))
    st.caption("RBF kernel이 유도하는 nonlinear feature space에서 SIR을 수행한 결과입니다.")
with d3:
    st.pyplot(scatter_plot(Z_q, y, f"Quantum Kernel SIR, {metric_name}={score_q:.3f}", task_type))
    st.caption("Qiskit quantum kernel이 유도하는 feature space에서 SIR을 수행한 결과입니다.")

st.subheader("7.6 Downstream Evaluation")
st.markdown(
    r"""
    각 2D embedding 위에서 간단한 supervised model을 학습해 성능을 비교합니다.

    - Classification: Logistic Regression Accuracy
    - Regression: Linear Regression \(R^2\)

    이 값은 embedding이 \(Y\)-관련 정보를 얼마나 잘 보존했는지 보는 참고 지표입니다.
    """
)

st.table(
    {
        "Method": ["Linear SIR", "RBF Kernel SIR", "Quantum Kernel SIR"],
        metric_name: [score_linear, score_rbf, score_q],
        "Leading eigenvalues": [
            np.round(eig_linear[:3], 4),
            np.round(eig_rbf[:3], 4),
            np.round(eig_q[:3], 4),
        ],
    }
)

scores = {
    "Linear SIR": score_linear,
    "RBF Kernel SIR": score_rbf,
    "Quantum Kernel SIR": score_q,
}
best_method = max(scores, key=scores.get)

st.info(
    f"""
    In this particular run, the best downstream score is obtained by **{best_method}**.  
    This result depends on dataset, noise level, sample size, kernel parameter, and quantum feature map depth.
    """
)

st.subheader("7.7 Quantum Feature Map Circuit")
st.markdown(
    r"""
    아래 회로는 classical data \(x\)를 quantum state \(|\phi(x)\rangle\)로 encoding하는 데 사용됩니다.  
    이 회로가 quantum kernel matrix \(K_Q\)를 정의합니다.
    """
)
st.text(feature_map.draw(output="text"))

st.subheader("8. Final Interpretation")
st.markdown("이번 실험은 다음 질문을 확인하기 위한 것입니다.")
st.latex(r"""\text{Can a quantum kernel be inserted into a Kernel SIR pipeline?}""")

st.markdown("이 앱에서 Quantum Kernel SIR은 다음 계산을 수행합니다.")
st.latex(
    r"""
    X
    \longrightarrow
    K_Q
    \longrightarrow
    \widehat M_{\mathrm{KSIR}}
    \longrightarrow
    Z_Q
    """
)

st.markdown("여기서 quantum kernel은 다음과 같이 정의됩니다.")
st.latex(
    r"""
    K_Q(x_i,x_j)
    =
    |\langle\phi(x_i)\mid\phi(x_j)\rangle|^2
    """
)

st.markdown(
    r"""
    이 프로젝트의 결론은 다음과 같습니다.

    1. Quantum kernel은 Kernel SIR pipeline에 실제로 삽입할 수 있습니다.
    2. Quantum Kernel SIR은 Linear SIR, RBF Kernel SIR과 같은 형식으로 비교할 수 있습니다.
    3. 그러나 Quantum Kernel SIR이 항상 가장 좋은 것은 아닙니다.
    4. 성능은 dataset geometry, noise, sample size, RBF gamma, `ZZFeatureMap reps`에 민감합니다.
    5. 따라서 이 프로젝트는 quantum advantage의 증명이 아니라, quantum kernel을 statistical dimension reduction에 접목한 exploratory implementation입니다.
    """
)
