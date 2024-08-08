import streamlit as st
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import Dict
from numba.types import int64, float64

# 状態とインデックスの対応を作成
state_list = ['H', 'LR1', 'LR2', 'HR', 'DA', 'DAU', 'DB', 'DBU', 'DC', 'DCU', 'DD', 'DDU', 'D', 'PRH', 'PDAH', 'PDBH', 'PDCH', 'PDDH']
state_map = Dict.empty(key_type=int64, value_type=int64)
for i, state in enumerate(state_list):
    state_map[i] = i

# データのアップロード
uploaded_file = st.file_uploader("senni_pro.csv", type="csv")
if uploaded_file is not None:
    change_probability = pd.read_csv(uploaded_file).to_numpy()

    # 初期状態を設定
    v_M_1 = np.array([state_map[0]] * 10000)  # everyone begins in the healthy state
    n_i = 10000
    n_t = 44
    d_c = d_e = 0.02  # コストとQALYsの割引率

    # コストの設定
    c_ma_f1_dukesA = 1786482
    c_ma_f1_dukesB = 2056922
    c_ma_f1_dukesC = 2637803
    c_ma_f1_dukesD = 3179764
    c_ma_f2_dukesA = 34610
    c_ma_f2_dukesB = 34610
    c_ma_f2_dukesC = 43758
    c_ma_f2_dukesD = 2476258

    # ユーティリティの設定
    u_dukeA = 0.879
    u_dukeB = 0.879
    u_dukeC = 0.867
    u_dukeD = 0.844
    u_normal = 1.0

    @njit
    def Probs(M_it, stage, dur, change_probability, state_map):
        n_s = 18  # number of states
        p_DA_sy = 0.065
        p_DB_sy = 0.26
        p_DC_sy = 0.46
        p_DD_sy = 0.92
        v_p_it = np.zeros(n_s)
        if M_it == state_map[0]:  # 'H'
            v_p_it[state_map[0]] = 1 - change_probability[stage, 1]
            v_p_it[state_map[1]] = change_probability[stage, 1]
        elif M_it == state_map[1]:  # 'LR1'
            v_p_it[state_map[1]] = 1 - change_probability[stage, 2]
            v_p_it[state_map[2]] = change_probability[stage, 2]
        elif M_it == state_map[2]:  # 'LR2'
            v_p_it[state_map[2]] = 1 - change_probability[stage, 3]
            v_p_it[state_map[3]] = change_probability[stage, 3]
        elif M_it == state_map[3]:  # 'HR'
            v_p_it[state_map[3]] = 1 - change_probability[stage, 4]
            v_p_it[state_map[4]] = change_probability[stage, 4]
        elif M_it == state_map[4]:  # 'DA'
            v_p_it[state_map[4]] = (1 - p_DA_sy) * (1 - change_probability[stage, 5] - change_probability[stage, 6])
            v_p_it[state_map[5]] = p_DA_sy * (1 - change_probability[stage, 6])
            v_p_it[state_map[6]] = (1 - p_DA_sy) * change_probability[stage, 5]
            v_p_it[state_map[12]] = change_probability[stage, 6]  # 'D'
        elif M_it == state_map[5]:  # 'DAU'
            if dur >= 4:
                v_p_it[state_map[14]] = 1  # 'PDAH'
            else:
                v_p_it[state_map[5]] = 1 - change_probability[stage, 6]
                v_p_it[state_map[12]] = change_probability[stage, 6]
        elif M_it == state_map[6]:  # 'DB'
            v_p_it[state_map[6]] = (1 - p_DB_sy) * (1 - change_probability[stage, 7] - change_probability[stage, 8])
            v_p_it[state_map[7]] = p_DB_sy * (1 - change_probability[stage, 8])
            v_p_it[state_map[8]] = (1 - p_DB_sy) * change_probability[stage, 7]
            v_p_it[state_map[12]] = change_probability[stage, 8]
        elif M_it == state_map[7]:  # 'DBU'
            if dur >= 4:
                v_p_it[state_map[15]] = 1  # 'PDBH'
            else:
                v_p_it[state_map[7]] = 1 - change_probability[stage, 8]
                v_p_it[state_map[12]] = change_probability[stage, 8]
        elif M_it == state_map[8]:  # 'DC'
            v_p_it[state_map[8]] = (1 - p_DC_sy) * (1 - change_probability[stage, 9] - change_probability[stage, 10])
            v_p_it[state_map[9]] = p_DC_sy * (1 - change_probability[stage, 10])
            v_p_it[state_map[10]] = (1 - p_DC_sy) * change_probability[stage, 9]
            v_p_it[state_map[12]] = change_probability[stage, 10]
        elif M_it == state_map[9]:  # 'DCU'
            if dur >= 4:
                v_p_it[state_map[16]] = 1  # 'PDCH'
            else:
                v_p_it[state_map[9]] = 1 - change_probability[stage, 10]
                v_p_it[state_map[12]] = change_probability[stage, 10]
        elif M_it == state_map[10]:  # 'DD'
            v_p_it[state_map[10]] = (1 - p_DD_sy) * (1 - change_probability[stage, 11])
            v_p_it[state_map[11]] = p_DD_sy * (1 - change_probability[stage, 11])
            v_p_it[state_map[12]] = change_probability[stage, 11]
        elif M_it == state_map[11]:  # 'DDU'
            if dur >= 4:
                v_p_it[state_map[17]] = 1  # 'PDDH'
            else:
                v_p_it[state_map[11]] = 1 - change_probability[stage, 11]
                v_p_it[state_map[12]] = change_probability[stage, 11]
        elif M_it == state_map[12]:  # 'D'
            v_p_it[state_map[12]] = 1
        elif M_it == state_map[13]:  # 'PRH'
            v_p_it[state_map[1]] = change_probability[stage, 12]
            v_p_it[state_map[2]] = change_probability[stage, 13]
            v_p_it[state_map[3]] = change_probability[stage, 14]
            v_p_it[state_map[13]] = 1 - change_probability[stage, 12] - change_probability[stage, 13] - change_probability[stage, 14]
        elif M_it == state_map[14]:  # 'PDAH'
            v_p_it[state_map[4]] = change_probability[stage, 15]
            v_p_it[state_map[14]] = 1 - change_probability[stage, 15]
        elif M_it == state_map[15]:  # 'PDBH'
            v_p_it[state_map[6]] = change_probability[stage, 16]
            v_p_it[state_map[15]] = 1 - change_probability[stage, 16]
        elif M_it == state_map[16]:  # 'PDCH'
            v_p_it[state_map[8]] = change_probability[stage, 17]
            v_p_it[state_map[16]] = 1 - change_probability[stage, 17]
        elif M_it == state_map[17]:  # 'PDDH'
            v_p_it[state_map[10]] = change_probability[stage, 18]
            v_p_it[state_map[17]] = 1 - change_probability[stage, 18]

        assert np.isclose(v_p_it.sum(), 1, atol=1e-6), "Probabilities do not sum to 1"
        return v_p_it

    @njit
    def Costs(M_it, next_M_it, p_it, state_map):
        c_it = np.zeros_like(M_it, dtype=np.float64)
        # DA -> DAU または症状ありで死亡する場合のコスト
        mask_DA_to_DAU = (M_it == state_map[4]) & (next_M_it == state_map[5])
        if np.any(mask_DA_to_DAU):
            c_it[mask_DA_to_DAU] = c_ma_f1_dukesA
        
        mask_DA_to_D = (M_it == state_map[4]) & (next_M_it == state_map[12]) & (p_it[:, 5] > 0)
        if np.any(mask_DA_to_D):
            c_it[mask_DA_to_D] = c_ma_f1_dukesA * p_it[mask_DA_to_D, 5]
        
        # DB -> DBU または症状ありで死亡する場合のコスト
        mask_DB_to_DBU = (M_it == state_map[6]) & (next_M_it == state_map[7])
        if np.any(mask_DB_to_DBU):
            c_it[mask_DB_to_DBU] = c_ma_f1_dukesB
        
        mask_DB_to_D = (M_it == state_map[6]) & (next_M_it == state_map[12]) & (p_it[:, 7] > 0)
        if np.any(mask_DB_to_D):
            c_it[mask_DB_to_D] = c_ma_f1_dukesB * p_it[mask_DB_to_D, 7]
        
        # DC -> DCU または症状ありで死亡する場合のコスト
        mask_DC_to_DCU = (M_it == state_map[8]) & (next_M_it == state_map[9])
        if np.any(mask_DC_to_DCU):
            c_it[mask_DC_to_DCU] = c_ma_f1_dukesC
        
        mask_DC_to_D = (M_it == state_map[8]) & (next_M_it == state_map[12]) & (p_it[:, 9] > 0)
        if np.any(mask_DC_to_D):
            c_it[mask_DC_to_D] = c_ma_f1_dukesC * p_it[mask_DC_to_D, 9]
        
        # DD -> DDU または症状ありで死亡する場合のコスト
        mask_DD_to_DDU = (M_it == state_map[10]) & (next_M_it == state_map[11])
        if np.any(mask_DD_to_DDU):
            c_it[mask_DD_to_DDU] = c_ma_f1_dukesD
        
        mask_DD_to_D = (M_it == state_map[10]) & (next_M_it == state_map[12]) & (p_it[:, 11] > 0)
        if np.any(mask_DD_to_D):
            c_it[mask_DD_to_D] = c_ma_f1_dukesD * p_it[mask_DD_to_D, 11]

        c_it[M_it == state_map[5]] = c_ma_f2_dukesA  # DAU
        c_it[M_it == state_map[7]] = c_ma_f2_dukesB  # DBU
        c_it[M_it == state_map[9]] = c_ma_f2_dukesC  # DCU
        c_it[M_it == state_map[11]] = c_ma_f2_dukesD  # DDU
    
        return c_it

    @njit
    def Effs(M_it, dur, state_map, X=None):
        if X is None:
            X = np.ones_like(M_it, dtype=np.float64)  # デフォルト値として1の配列を使用
        u_it = np.zeros_like(M_it, dtype=np.float64)
        u_it[M_it == state_map[0]] = u_normal
        u_it[M_it == state_map[1]] = u_normal
        u_it[M_it == state_map[2]] = u_normal
        u_it[M_it == state_map[3]] = u_normal
        u_it[M_it == state_map[4]] = u_dukeA
        u_it[M_it == state_map[5]] = u_dukeA
        u_it[M_it == state_map[6]] = u_dukeB
        u_it[M_it == state_map[7]] = u_dukeB
        u_it[M_it == state_map[8]] = u_dukeC
        u_it[M_it == state_map[9]] = u_dukeC
        u_it[M_it == state_map[10]] = u_dukeD
        u_it[M_it == state_map[11]] = u_dukeD
        u_it[M_it == state_map[12]] = 0
        u_it[M_it == state_map[13]] = u_normal
        u_it[M_it == state_map[14]] = u_normal
        u_it[M_it == state_map[15]] = u_normal
        u_it[M_it == state_map[16]] = u_normal
        u_it[M_it == state_map[17]] = u_normal
        return u_it * X

    @njit(parallel=True)
    def samplev(probs, m):
        n, k = probs.shape  # nは1、kはマルコフ状態の数、今回なら18が入る
        ran = np.empty((n, m), dtype=np.int64)  # サンプリング結果を格納する配列
        for i in prange(n):
            u = np.random.random(m)  # 各個人ごとにランダムな値を生成
            cumulative_sum = np.zeros(k)
            cumulative_sum[0] = probs[i, 0]
            for j in range(1, k):
                cumulative_sum[j] = cumulative_sum[j - 1] + probs[i, j]
            for j in range(m):
                ran[i, j] = np.searchsorted(cumulative_sum, u[j])
        return ran

    def MicroSim(v_M_1, n_i, n_t, states, change_probability, state_map, X=None, d_c=0.02, d_e=0.02, TR_out=True, TS_out=True, Trt=False, seed=1):
        np.random.seed(seed)
        v_dwc = 1 / (1 + d_c) ** np.arange(n_t + 1)  # コストの割引率
        v_dwe = 1 / (1 + d_e) ** np.arange(n_t + 1)  # QALYの割引率

        m_M = np.empty((n_i, n_t + 1), dtype=np.int32)
        m_C = np.zeros((n_i, n_t + 1))
        m_E = np.zeros((n_i, n_t + 1))

        m_M[:, 0] = v_M_1  # 初期状態を設定
        dur = np.zeros(n_i, dtype=np.int32)  # 病気の期間を初期化
        m_C[:, 0] = Costs(m_M[:, 0], m_M[:, 0], np.zeros((n_i, 18)), state_map)
        m_E[:, 0] = Effs(m_M[:, 0], dur, state_map=state_map, X=X)
        
        for t in range(1, n_t + 1):
            m_p = np.array([Probs(state, t, dur[i], change_probability, state_map) for i, state in enumerate(m_M[:, t - 1])])
            next_states = np.array([states[i] for i in samplev(m_p, 1).flatten()])
            m_C[:, t] = Costs(m_M[:, t - 1], next_states, m_p, state_map)
            m_M[:, t] = next_states
            m_E[:, t] = Effs(m_M[:, t], dur, state_map=state_map, X=X)
            dur = np.where(np.isin(m_M[:, t], [state_map[5], state_map[7], state_map[9], state_map[11]]), dur + 1, 0)
            if t % 10 == 0:
                st.write(f"{t / n_t * 100:.0f}% done")

        tc = m_C @ v_dwc  # 割引後の総コスト
        te = m_E @ v_dwe  # 割引後の総QALYs
        tc_hat = np.mean(tc)  # 平均コスト
        te_hat = np.mean(te)  # 平均QALYs
        
        if TS_out:
            TS = pd.DataFrame(
                {f"Cycle_{i}": [state_map[m_M[j, i]] for j in range(n_i)] for i in range(n_t)}
            )
        else:
            TS = None

        if TR_out:
            TR = pd.DataFrame(m_M).apply(pd.Series.value_counts).fillna(0).T / n_i
            TR = TR.reindex(columns=list(state_map.values()), fill_value=0)
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

    # シミュレーションの実行
    sim_no_trt_with_TR = MicroSim(v_M_1, n_i, n_t, list(state_map.values()), change_probability, state_map, X=None, d_c=d_c, d_e=d_e, Trt=False, TR_out=True, seed=100)

    # 結果の表示
    if sim_no_trt_with_TR["TR"] is not None:
        st.write("Transition probabilities (with TR_out=True):")
        st.write(sim_no_trt_with_TR["TR"])
    else:
        st.write("TR not calculated.")

    # シミュレーションの実行
    import time

    start_time = time.time()

    # No Treatment戦略のシミュレーション
    sim_no_trt = MicroSim(v_M_1, n_i, n_t, list(state_map.values()), change_probability, state_map, X=None, d_c=d_c, d_e=d_e, TR_out=True, TS_out=True, Trt=False, seed=100)

    # Treatment戦略のシミュレーション
    sim_trt = MicroSim(v_M_1, n_i, n_t, list(state_map.values()), change_probability, state_map, X=None, d_c=d_c, d_e=d_e, TR_out=True, TS_out=True, Trt=True, seed=100)

    comp_time = time.time() - start_time
    st.write(f"Computation time: {comp_time:.2f} seconds")

    # 平均コストと平均QALYsを表示
    st.write("No Treatment strategy:")
    st.write(f"Average cost: {sim_no_trt['tc_hat']}")
    st.write(f"Average QALYs: {sim_no_trt['te_hat']}")

    st.write("\nTreatment strategy:")
    st.write(f"Average cost: {sim_trt['tc_hat']}")
    st.write(f"Average QALYs: {sim_trt['te_hat']}")
    st.write(sim_trt['tc'].sum())

else:
    st.write("Please upload a CSV file.")
