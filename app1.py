import streamlit as st
import numpy as np
import pandas as pd
from numba import njit

# データの読み込み
change_probability = pd.read_csv('senni_pro.csv')
variable = pd.read_csv('Variable.csv')
markov_state = pd.read_csv('markov_state.csv')
test_quality = pd.read_csv('test_quality.csv')
symptom = pd.read_csv('symptom.csv')
cost = pd.read_csv('cost.csv')
utility = pd.read_csv('utility.csv')
setting = pd.read_csv('setting.csv')
accident_rate = pd.read_csv('accident_rate.csv')

# Model input
n_i = 10000  # number of simulated individuals
n_t = 44  # time horizon, 30 cycles
# markov_stateの2列目の値を取得し、1行目を除いてリスト化
v_n = markov_state.iloc[0:, 1].tolist()
n_s = len(v_n)  # the number of health states
v_M_1 = ["H"] * n_i  # everyone begins in the healthy state
d_c = d_e = 0.02  # equal discounting of costs and QALYs by 3%
v_Trt = ["No Treatment", "Treatment"]  # store the strategy names

@njit
def Probs(M_it, stage,dur):
    """指定されたステージのマルコフ状態に対する遷移確率を計算する."""
    # ステージごとの遷移確率を取得
    p_normal_to_lr1 = change_probability.iloc[stage, 1]
    p_lr1_to_lr2 = change_probability.iloc[stage, 2]
    p_lr2_to_hr = change_probability.iloc[stage, 3]
    p_hr_to_da = change_probability.iloc[stage, 4]
    p_da_to_db = change_probability.iloc[stage, 5]
    p_da_to_death = change_probability.iloc[stage, 6]
    p_db_to_dc = change_probability.iloc[stage,7]
    p_db_to_death = change_probability.iloc[stage, 8]
    p_dc_to_dd = change_probability.iloc[stage, 9]
    p_dc_to_death = change_probability.iloc[stage, 10]
    p_dd_to_death = change_probability.iloc[stage, 11]
    p_prn_to_lr1 = change_probability.iloc[stage, 12]
    p_prn_to_lr2 = change_probability.iloc[stage, 13]
    p_prn_to_hr = change_probability.iloc[stage, 14]
    p_r_dukeA = change_probability.iloc[stage, 15]
    p_r_dukeB = change_probability.iloc[stage, 16]
    p_r_dukeC = change_probability.iloc[stage, 17]    
    p_r_dukeD = change_probability.iloc[stage, 18]
    p_DA_sy   = 0.065 #DukeAで症状がある割合
    p_DB_sy   = 0.26
    p_DC_sy   = 0.46
    p_DD_sy   = 0.92

    #DAU, DBU, DCU, DDUに関してはマイクロシミュレーション的に分類させる必要がある。
    v_p_it = np.zeros(n_s)
    if M_it == "H":
        v_p_it[:] = [1 - p_normal_to_lr1, p_normal_to_lr1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif M_it == "LR1":
        v_p_it[:] = [0, 1 - p_lr1_to_lr2, p_lr1_to_lr2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif M_it == "LR2":
        v_p_it[:] = [0, 0, 1 - p_lr2_to_hr, p_lr2_to_hr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif M_it == "HR":
        v_p_it[:] = [0, 0, 0, 1 - p_hr_to_da, p_hr_to_da, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    elif M_it == "DA":
        v_p_it[:] = [0, 0, 0, 0, (1-p_DA_sy)*(1-p_da_to_db-p_da_to_death),p_DA_sy*(1-p_da_to_death),(1-p_DA_sy)*p_da_to_db, 0, 0, 0, 0, 0, p_da_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DAU":
        if dur >= 4:  # 4回続いた場合
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        else:
            v_p_it[:] = [0, 0, 0, 0, 0, 1-p_da_to_death, 0, 0, 0, 0, 0, 0, p_da_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DBU":
        if dur >= 4:  # 4回続いた場合
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        else:
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 1 - p_db_to_death, 0, 0, 0, 0, p_db_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DCU":
        if dur >= 4:  # 4回続いた場合
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        else:
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - p_dc_to_death, 0, 0, p_dc_to_death, 0, 0, 0, 0, 0]

    elif M_it == "DDU":
        if dur >= 4:  # 4回続いた場合
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - p_dd_to_death, p_dd_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DB":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, (1-p_DB_sy)*(1-p_db_to_dc-p_db_to_death), p_DB_sy*(1-p_db_to_death), (1-p_DB_sy)*p_db_to_dc, 0, 0, 0, p_db_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DC":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, (1 - p_DC_sy)*(1 - p_dc_to_dd- p_dc_to_death), p_DC_sy*(1-p_dc_to_death), (1 - p_DC_sy)*p_dc_to_dd, 0, p_dc_to_death, 0, 0, 0, 0, 0]
    elif M_it == "DD":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1 - p_DD_sy)*(1 - p_dd_to_death), p_DD_sy*(1 - p_dd_to_death), p_dd_to_death, 0, 0, 0, 0, 0]
    elif M_it == "D":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    elif M_it == "PRH":
        v_p_it[:] = [0, p_prn_to_lr1, p_prn_to_lr2, p_prn_to_hr,0 , 0, 0, 0, 0, 0, 0, 0, 0, 1 - p_prn_to_lr1 - p_prn_to_lr2 - p_prn_to_hr, 0, 0, 0, 0]
    elif M_it == "PDAH":
        v_p_it[:] = [0, 0, 0, 0, p_r_dukeA, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1-p_r_dukeA, 0, 0, 0]
    elif M_it == "PDBH":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, p_r_dukeB, 0, 0, 0, 0, 0, 0, 0, 0, 1-p_r_dukeB, 0, 0]
    elif M_it == "PDCH":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, p_r_dukeC, 0, 0, 0, 0, 0, 0, 0, 1-p_r_dukeC, 0]
    elif M_it == "PDDH":
        v_p_it[:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_r_dukeD, 0, 0, 0, 0, 0, 0, 1-p_r_dukeD]
   # 出力の確認
    assert np.isclose(v_p_it.sum(), 1, atol=1e-6), "Probabilities do not sum to 1"
    return v_p_it

# コストデータの設定
c_fit = 1557
c_ma_f1_dukesA = 1786482
c_ma_f1_dukesB = 2056922
c_ma_f1_dukesC = 2637803
c_ma_f1_dukesD = 3179764
c_ma_f2_dukesA = 34610
c_ma_f2_dukesB = 34610
c_ma_f2_dukesC = 43758
c_ma_f2_dukesD = 2476258
c_rem_hr_ply = 452445
c_rem_lr_poly = 48650
c_tcs = 30943

@njit
def Costs(M_it, next_M_it, p_it, Trt=False):
    """各マルコフ状態に対応するコストを計算する関数"""
    c_it = np.zeros_like(M_it, dtype=float)
    c_it[M_it == "H"] = 0
    c_it[M_it == "LR1"] = 0
    c_it[M_it == "LR2"] = 0
    c_it[M_it == "HR"] = 0
    # c_it[M_it == "DA"] = c_ma_f1_dukesA
    c_it[M_it == "DAU"] = c_ma_f2_dukesA
    # c_it[M_it == "DB"] = c_ma_f1_dukesB
    c_it[M_it == "DBU"] = c_ma_f2_dukesB
    # c_it[M_it == "DC"] = c_ma_f1_dukesC
    c_it[M_it == "DCU"] = c_ma_f2_dukesC
    # c_it[M_it == "DD"] = c_ma_f1_dukesD
    c_it[M_it == "DDU"] = c_ma_f2_dukesD
    c_it[M_it == "D"] = 0  # 死亡状態にはコストなし
    c_it[M_it == "PRH"] = 0  # 仮にコストなしと設定
    c_it[M_it == "PDAH"] = 0  # 仮にコストなしと設定
    c_it[M_it == "PDBH"] = 0  # 仮にコストなしと設定
    c_it[M_it == "PDCH"] = 0  # 仮にコストなしと設定
    c_it[M_it == "PDDH"] = 0  # 仮にコストなしと設定
    # DA -> DAU または症状ありで死亡する場合のコスト
    mask_DA_to_DAU = (M_it == "DA") & (next_M_it == "DAU")
    if np.any(mask_DA_to_DAU):
        c_it[mask_DA_to_DAU] = c_ma_f1_dukesA
    
    mask_DA_to_D = (M_it == "DA") & (next_M_it == "D") & (p_it[:, 5] > 0)
    if np.any(mask_DA_to_D):
        c_it[mask_DA_to_D] = c_ma_f1_dukesA * p_it[mask_DA_to_D, 5]
    
    # DB -> DBU または症状ありで死亡する場合のコスト
    mask_DB_to_DBU = (M_it == "DB") & (next_M_it == "DBU")
    if np.any(mask_DB_to_DBU):
        c_it[mask_DB_to_DBU] = c_ma_f1_dukesB
    
    mask_DB_to_D = (M_it == "DB") & (next_M_it == "D") & (p_it[:, 7] > 0)
    if np.any(mask_DB_to_D):
        c_it[mask_DB_to_D] = c_ma_f1_dukesB * p_it[mask_DB_to_D, 7]
    
    # DC -> DCU または症状ありで死亡する場合のコスト
    mask_DC_to_DCU = (M_it == "DC") & (next_M_it == "DCU")
    if np.any(mask_DC_to_DCU):
        c_it[mask_DC_to_DCU] = c_ma_f1_dukesC
    
    mask_DC_to_D = (M_it == "DC") & (next_M_it == "D") & (p_it[:, 9] > 0)
    if np.any(mask_DC_to_D):
        c_it[mask_DC_to_D] = c_ma_f1_dukesC * p_it[mask_DC_to_D, 9]
    
    # DD -> DDU または症状ありで死亡する場合のコスト
    mask_DD_to_DDU = (M_it == "DD") & (next_M_it == "DDU")
    if np.any(mask_DD_to_DDU):
        c_it[mask_DD_to_DDU] = c_ma_f1_dukesD
    
    mask_DD_to_D = (M_it == "DD") & (next_M_it == "D") & (p_it[:, 11] > 0)
    if np.any(mask_DD_to_D):
        c_it[mask_DD_to_D] = c_ma_f1_dukesD * p_it[mask_DD_to_D, 11]
    return c_it

# ユーティリティデータの設定
u_dukeA = utility.loc[0, 'value']
u_dukeB = utility.loc[1, 'value']
u_dukeC = utility.loc[2, 'value']
u_dukeD = utility.loc[3, 'value']
u_hl    = utility.loc[4, 'value']
u_lr1   = utility.loc[5, 'value']
u_lr2   = utility.loc[6, 'value']
u_no_sy = utility.loc[7, 'value']
u_normal= utility.loc[8, 'value']

@njit
def Effs(M_it, dur, Trt=False, cl=1, X=None):
    """各マルコフ状態に対応するユーティリティを計算する関数"""
    if X is None:
        X = np.ones_like(M_it, dtype=float)  # デフォルト値として1の配列を使用
    u_it = np.zeros_like(M_it, dtype=float)
    u_it[M_it == "H"] = u_normal
    u_it[M_it == "LR1"] = u_normal
    u_it[M_it == "LR2"] = u_normal
    u_it[M_it == "HR"] = u_normal
    u_it[M_it == "DA"] = u_dukeA
    u_it[M_it == "DAU"] = u_dukeA
    u_it[M_it == "DB"] = u_dukeB
    u_it[M_it == "DBU"] = u_dukeB
    u_it[M_it == "DC"] = u_dukeC
    u_it[M_it == "DCU"] = u_dukeC
    u_it[M_it == "DD"] = u_dukeD
    u_it[M_it == "DDU"] = u_dukeD
    u_it[M_it == "D"] = 0
    u_it[M_it == "PRH"] = u_normal
    u_it[M_it == "PDAH"] = u_normal
    u_it[M_it == "PDBH"] = u_normal
    u_it[M_it == "PDCH"] = u_normal
    u_it[M_it == "PDDH"] = u_normal
    return u_it * cl

@njit
def samplev(probs, m):
    n, k = probs.shape  # nは1、kはマルコフ状態の数、今回なら18が入る
    ran = np.empty((n, m), dtype=int)  # サンプリング結果を格納する配列
    U = np.cumsum(probs, axis=1)  # 横方向に累積和を計算
    if not np.allclose(U[:, -1], 1):
        raise ValueError("Probabilities do not sum to 1")

    for j in range(m):
        un = np.random.rand(n, 1)  # 各個人ごとにランダムな値を生成
        ran[:, j] = (un > U).sum(axis=1)  # サンプリングした値が累積確率を超える位置を求める、ranは1行18列
    return ran

def MicroSim(v_M_1, n_i, n_t, states, X=None, d_c=0.02, d_e=0.02, TR_out=True, TS_out=True, Trt=False, seed=1):
    np.random.seed(seed)
    v_dwc = 1 / (1 + d_c) ** np.arange(n_t + 1)  # コストの割引率
    v_dwe = 1 / (1 + d_e) ** np.arange(n_t + 1)  # QALYの割引率

    m_M = np.empty((n_i, n_t + 1), dtype='U4')
    m_C = np.zeros((n_i, n_t + 1))
    m_E = np.zeros((n_i, n_t + 1))

    m_M[:, 0] = v_M_1  # 初期状態を設定
    dur = np.zeros(n_i)  # 病気の期間を初期化
    m_C[:, 0] = Costs(m_M[:, 0], m_M[:, 0], np.zeros((n_i, n_s)), Trt)
    m_E[:, 0] = Effs(m_M[:, 0], dur, Trt, X=X)
    
    for t in range(1, n_t + 1):
        m_p = np.array([Probs(state, t, dur[i]) for i, state in enumerate(m_M[:, t - 1])])
        next_states = np.array([states[i] for i in samplev(m_p, 1).flatten()])
        m_C[:, t] = Costs(m_M[:, t - 1], next_states, m_p, Trt)
        m_M[:, t] = next_states
        m_E[:, t] = Effs(m_M[:, t], dur, Trt, X=X)
        dur = np.where(np.isin(m_M[:, t], ["DAU", "DBU", "DCU", "DDU"]), dur + 1, 0)
        if t % 10 == 0:
            print(f"\r{t / n_t * 100:.0f}% done", end="")

    tc = m_C @ v_dwc  # 割引後の総コスト
    te = m_E @ v_dwe  # 割引後の総QALYs
    tc_hat = np.mean(tc)  # 平均コスト
    te_hat = np.mean(te)  # 平均QALYs

    if TS_out:
        TS = pd.DataFrame(
            {f"Cycle_{i}": [f"{m_M[j, i]}->{m_M[j, i+1]}" for j in range(n_i)] for i in range(n_t)}
        )
    else:
        TS = None

    if TR_out:
        TR = pd.DataFrame(m_M).apply(pd.Series.value_counts).fillna(0).T / n_i
        TR = TR.reindex(columns=states, fill_value=0)
    else:
        TR = None

    results = {
        "m_M": m_M,
        "m_C": m_C,
        "m_E": m_E,
        "tc": tc,
        "te": te,
        "tc_hat": tc_hat,
        "te_hat": te_hat,
        "TS": TS,
        "TR": TR
    }
    return results

# Streamlit UI
st.title("Colon Cancer Screening Cost-Effectiveness Analysis")

# シミュレーションの実行 (TR_out=True)
st.write("Running No Treatment strategy simulation...")
sim_no_trt_with_TR = MicroSim(v_M_1, n_i, n_t, v_n, X=None, d_c=d_c, d_e=d_e, Trt=False, TR_out=True, seed=100)

# 結果の表示
if sim_no_trt_with_TR["TR"] is not None:
    st.write("Transition probabilities (with TR_out=True):")
    st.dataframe(sim_no_trt_with_TR["TR"])
else:
    st.write("TR not calculated.")

# シミュレーションの実行
import time

start_time = time.time()

# No Treatment戦略のシミュレーション
st.write("Running No Treatment strategy simulation...")
sim_no_trt = MicroSim(v_M_1, n_i, n_t, v_n, X=None, d_c=d_c, d_e=d_e, TR_out=True, TS_out=True, Trt=False, seed=100)

# Treatment戦略のシミュレーション
st.write("Running Treatment strategy simulation...")
sim_trt = MicroSim(v_M_1, n_i, n_t, v_n, X=None, d_c=d_c, d_e=d_e, TR_out=True, TS_out=True, Trt=True, seed=100)

comp_time = time.time() - start_time
st.write(f"Computation time: {comp_time:.2f} seconds")

# 平均コストと平均QALYsを表示
st.write("No Treatment strategy:")
st.write(f"Average cost: {sim_no_trt['tc_hat']}")
st.write(f"Average QALYs: {sim_no_trt['te_hat']}")

st.write("\nTreatment strategy:")
st.write(f"Average cost: {sim_trt['tc_hat']}")
st.write(f"Average QALYs: {sim_trt['te_hat']}")
st.write(f"Total cost: {sim_trt['tc'].sum()}")
