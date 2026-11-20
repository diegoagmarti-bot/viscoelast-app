"""Viscoelasticidade linear unidimensional: aplicativo Streamlit."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

import modelos as m
from esquema import COR_AMORT, COR_MOLA, figura_animada

st.set_page_config(page_title="Viscoelasticidade Linear 1D", page_icon=":material/timeline:",
                   layout="wide")

ENSAIOS = {
    "Fluência (tensão constante)": "fluencia",
    "Relaxação (deformação constante)": "relaxacao",
    "Fluência e recuperação": "recuperacao",
    "Histórico de tensão personalizado": "personalizado",
    "Carga senoidal (DMTA)": "senoidal",
}
HISTORICO_PADRAO = pd.DataFrame({"t (s)": [0.0, 5.0, 15.0, 15.0, 30.0],
                                 "σ (MPa)": [0.0, 10.0, 10.0, 0.0, 0.0]})
CORES = [COR_MOLA, COR_AMORT, "#4F8A5B", "#7A5C99"]
GRADE = dict(showgrid=True, gridcolor="#E6E9EE")
N_PONTOS = 1200
ROT_EPS, ROT_SIG = "Deformação, ε (%)", "Tensão, σ (MPa)"


def grafico(height=360, **layout):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=20, t=80, b=50),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **layout)
    fig.update_xaxes(**GRADE)
    fig.update_yaxes(**GRADE)
    return fig


def valores_de(E_, eta_, Einf_, carga, ensaio, t_rem=None):
    """Dicionário símbolo -> valor numérico usado nas substituições."""
    v = {m.E: E_, m.eta: eta_, m.Einf: Einf_, m.tau: eta_ / E_}
    if ensaio in ("fluencia", "recuperacao"):
        v[m.sig0] = carga
    elif ensaio in ("relaxacao", "senoidal"):
        v[m.eps0] = carga
    if t_rem is not None:
        v[m.t1] = t_rem
    return v


@st.cache_data(show_spinner=False)
def calcular(nome, ensaio, E_, eta_, Einf_, carga, t_final, t_rem=None, omega=None, historico=None):
    """Tempo, tensão e deformação. Resultados ficam em cache entre interações."""
    valores = valores_de(E_, eta_, Einf_, carga, ensaio, t_rem)
    tt = np.linspace(0, t_final, N_PONTOS)
    sig, eps = m.simular(nome, ensaio, valores, tt, t_rem=t_rem, omega=omega, historico=historico)
    return tt, sig, eps


@st.cache_data(show_spinner=False)
def residual(nome, E_, eta_, Einf_, s0, t_rem):
    """Deformação residual (t -> infinito) após a remoção da carga na fluência e recuperação."""
    expr = m.solucao_degrau(nome, "fluencia")["saida_t"].subs(valores_de(E_, eta_, Einf_, s0, "fluencia"))
    return float(sp.limit(expr - expr.subs(m.t, m.t - t_rem), m.t, sp.oo))


def num(expr, valores):
    """Substitui os valores numéricos e arredonda para exibição."""
    return sp.N(expr.subs(valores), 4)


def fmt_lim(x, pct=False):
    x = sp.sympify(x)
    if x.is_infinite:
        return "ilimitada"
    return f"{float(x) * (100 if pct else 1):.4g}"


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuração")
    nome = st.selectbox("Modelo reológico", list(m.MODELOS))
    ensaio_rot = st.selectbox("Histórico de carregamento", list(ENSAIOS))
    ensaio = ENSAIOS[ensaio_rot]
    modelo = m.MODELOS[nome]

    st.subheader("Parâmetros do material")
    E_ = st.number_input("Módulo da mola, E (MPa)", min_value=1e-6, value=2000.0, format="%.4g")
    eta_ = st.number_input("Viscosidade do amortecedor, η (MPa·s)", min_value=1e-6, value=10000.0,
                           format="%.4g")
    Einf_ = st.number_input("Módulo da mola isolada, E∞ (MPa)", min_value=1e-6, value=1000.0,
                            format="%.4g", disabled=nome != "Sólido Linear Padrão",
                            help="Usado no Sólido Linear Padrão e na aba de comparação.")
    tau = eta_ / E_

    st.subheader("Carregamento")
    t_rem = omega = freq = historico = carga = None
    if ensaio in ("fluencia", "recuperacao"):
        carga = st.number_input("Tensão aplicada, σ0 (MPa)", min_value=1e-6, value=10.0, format="%.4g")
    elif ensaio in ("relaxacao", "senoidal"):
        carga = st.number_input("Amplitude de deformação, ε0 (%)", min_value=1e-7, value=0.5,
                                format="%.4g") / 100

    if ensaio == "senoidal":
        freq = st.number_input("Frequência, f (Hz)", min_value=1e-6, value=round(1 / (2 * np.pi * tau), 4),
                               format="%.4g", help="Valor inicial: ω = 1/τ.")
        ciclos = st.slider("Número de ciclos exibidos", 1, 10, 3)
        omega = 2 * np.pi * freq
        t_final = ciclos / freq
    elif ensaio == "personalizado":
        st.caption("Pontos (tempo, tensão) ligados por retas. Repita o tempo para criar um degrau.")
        tabela = st.data_editor(HISTORICO_PADRAO, num_rows="dynamic", hide_index=True, width="stretch")
        tabela = tabela.dropna().astype(float).sort_values("t (s)", kind="stable")
        tabela = tabela[tabela["t (s)"] >= 0]
        if len(tabela) < 2 or tabela["t (s)"].max() <= 0:
            st.error("Informe ao menos dois pontos, com tempos não negativos.")
            st.stop()
        historico = (tuple(tabela["t (s)"]), tuple(tabela["σ (MPa)"]))
        t_final = float(tabela["t (s)"].max())
    else:
        t_final = st.number_input("Tempo total de análise (s)", min_value=1e-6, value=round(6 * tau, 4),
                                  format="%.4g")

    if ensaio == "recuperacao":
        if st.session_state.get("t1", np.inf) > t_final:
            st.session_state["t1"] = t_final / 2
        t_rem = st.number_input("Instante de remoção da carga, t1 (s)", min_value=1e-6,
                                max_value=float(t_final), format="%.4g", key="t1")

# ---------------------------------------------------------------- simulação
valores = valores_de(E_, eta_, Einf_, carga, ensaio, t_rem)
tt, sig, eps = calcular(nome, ensaio, E_, eta_, Einf_, carga, t_final, t_rem, omega, historico)
comp = m.componentes(nome, sig, eps, valores)
controle_tensao = ensaio in ("fluencia", "recuperacao", "personalizado")
resposta, rot_resp = (eps * 100, ROT_EPS) if controle_tensao else (sig, ROT_SIG)

# ---------------------------------------------------------------- cabeçalho
st.title("Viscoelasticidade Linear Unidimensional")
st.caption(f"Modelo de {nome}  ·  {modelo['arranjo']}  ·  {ensaio_rot}")

st.markdown("**Equação constitutiva diferencial**")
st.latex(m.equacao_diferencial(nome) + r", \qquad \tau = \eta / E")
c1, c2, c3, c4 = st.columns(4)
if ensaio == "senoidal":
    sol_s = m.solucao_senoidal(nome)
    Ea = float(sol_s["E_arm"].subs(valores).subs(m.w, omega))
    Ep = float(sol_s["E_perda"].subs(valores).subs(m.w, omega))
    c1.metric("E′, armazenamento (MPa)", f"{Ea:.4g}", border=True)
    c2.metric("E″, perda (MPa)", f"{Ep:.4g}", border=True)
    c3.metric("tan δ", f"{Ep / Ea:.4g}", border=True)
    c4.metric("Ângulo de fase δ (graus)", f"{np.degrees(np.arctan2(Ep, Ea)):.4g}", border=True)
elif ensaio == "personalizado":
    c1.metric(modelo["tau_nome"], f"{tau:.4g}", border=True)
    c2.metric("σ máxima aplicada (MPa)", f"{np.max(sig):.4g}", border=True)
    c3.metric("ε máxima (%)", f"{np.max(eps) * 100:.4g}", border=True)
    c4.metric("ε no fim da análise (%)", f"{eps[-1] * 100:.4g}", border=True)
else:
    tipo = "relaxacao" if ensaio == "relaxacao" else "fluencia"
    sol_d = m.solucao_degrau(nome, tipo)
    pct = tipo == "fluencia"
    rot, un = ("σ", "MPa") if tipo == "relaxacao" else ("ε", "%")
    c1.metric(modelo["tau_nome"], f"{tau:.4g}", border=True)
    c2.metric(f"Resposta instantânea {rot}(0⁺) ({un})", fmt_lim(sol_d["inicial"].subs(valores), pct),
              border=True)
    c3.metric(f"{rot} no fim da análise ({un})", f"{resposta[-1]:.4g}", border=True)
    if ensaio == "recuperacao":
        c4.metric("ε residual, t → ∞ (%)", f"{residual(nome, E_, eta_, Einf_, carga, t_rem) * 100:.4g}",
                  border=True)
    else:
        c4.metric(f"{rot}(t → ∞) ({un})", fmt_lim(sol_d["final"].subs(valores), pct), border=True)

abas = [":material/animation: Simulação", ":material/stacked_line_chart: Componentes",
        ":material/compare: Comparação de modelos", ":material/function: Solução analítica"]
if ensaio == "senoidal":
    abas.append(":material/graphic_eq: Análise dinâmica (DMTA)")
tabs = st.tabs(abas)

# ---------------------------------------------------------------- aba: simulação
with tabs[0]:
    st.markdown("Use **Reproduzir** ou arraste o controle de tempo para ver molas e amortecedores "
                "se deformarem. Cada elemento é desenhado em sua própria escala, para que mesmo "
                "deformações pequenas fiquem visíveis.")
    forca = sig / (np.max(np.abs(sig)) or 1.0)
    fig = figura_animada(nome, tt, [c[0] for c in comp.values()], forca, resposta, rot_resp)
    st.plotly_chart(fig, width="stretch")
    if nome == "Kelvin-Voigt" and ensaio == "relaxacao":
        st.info("No modelo de Kelvin-Voigt, impor um degrau de deformação exige uma tensão impulsiva "
                "η ε0 δ(t) em t = 0, pois o amortecedor não pode se deformar instantaneamente. "
                "O gráfico mostra a resposta para t > 0.", icon=":material/info:")

# ---------------------------------------------------------------- aba: componentes
with tabs[1]:
    a, b = st.columns(2, gap="large")
    with a:
        f = grafico(title="Histórico aplicado e resposta")
        f.add_trace(go.Scatter(x=tt, y=sig, name="Tensão σ (MPa)", line=dict(color=COR_MOLA)))
        f.add_trace(go.Scatter(x=tt, y=eps * 100, name="Deformação ε (%)", yaxis="y2",
                               line=dict(color=COR_AMORT)))
        f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title=ROT_SIG,
                        yaxis2=dict(title=ROT_EPS, overlaying="y", side="right", showgrid=False))
        st.plotly_chart(f, width="stretch")
    with b:
        f = grafico(title="Deformação em cada componente")
        for (rot, (e_c, _)), cor in zip(comp.items(), CORES):
            f.add_trace(go.Scatter(x=tt, y=e_c * 100, name=rot, line=dict(color=cor)))
        f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title=ROT_EPS)
        st.plotly_chart(f, width="stretch")
    f = grafico(title="Tensão em cada componente")
    for (rot, (_, s_c)), cor in zip(comp.items(), CORES):
        f.add_trace(go.Scatter(x=tt, y=s_c, name=rot, line=dict(color=cor)))
    f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title=ROT_SIG)
    st.plotly_chart(f, width="stretch")
    st.caption("A mola responde instantaneamente (σ = E ε). O amortecedor responde à taxa de "
               "deformação (σ = η dε/dt). Em série, os elementos compartilham a tensão e somam as "
               "deformações; em paralelo, compartilham a deformação e somam as tensões.")

    dados = {"t (s)": tt, "sigma (MPa)": sig, "epsilon (mm/mm)": eps}
    for rot, (e_c, s_c) in comp.items():
        dados[f"epsilon {rot} (mm/mm)"] = e_c
        dados[f"sigma {rot} (MPa)"] = s_c
    st.download_button("Baixar dados (CSV)", pd.DataFrame(dados).to_csv(index=False).encode("utf-8"),
                       file_name="viscoelasticidade.csv", mime="text/csv", icon=":material/download:")

# ---------------------------------------------------------------- aba: comparação
with tabs[2]:
    st.markdown(f"Os três modelos sob o mesmo carregamento, com E = {E_:.6g} MPa, η = {eta_:.6g} MPa·s "
                f"e, no Sólido Linear Padrão, E∞ = {Einf_:.6g} MPa.")
    f = grafico(height=420, title="Resposta de cada modelo")
    linhas = []
    for nm, cor in zip(m.MODELOS, CORES):
        _, s_n, e_n = calcular(nm, ensaio, E_, eta_, Einf_, carga, t_final, t_rem, omega, historico)
        y = e_n * 100 if controle_tensao else s_n
        f.add_trace(go.Scatter(x=tt, y=y, name=nm, line=dict(color=cor, width=3 if nm == nome else 2)))
        if ensaio == "senoidal":
            sol_n = m.solucao_senoidal(nm)
            Ea_n = float(sol_n["E_arm"].subs(valores).subs(m.w, omega))
            Ep_n = float(sol_n["E_perda"].subs(valores).subs(m.w, omega))
            linhas.append({"Modelo": nm, "E′ (MPa)": f"{Ea_n:.4g}", "E″ (MPa)": f"{Ep_n:.4g}",
                           "tan δ": f"{Ep_n / Ea_n:.4g}"})
        elif ensaio != "personalizado":
            sol_n = m.solucao_degrau(nm, "relaxacao" if ensaio == "relaxacao" else "fluencia")
            un = "σ (MPa)" if ensaio == "relaxacao" else "ε (%)"
            linhas.append({"Modelo": nm,
                           f"Instantânea, {un}": fmt_lim(sol_n["inicial"].subs(valores), controle_tensao),
                           f"Longo prazo sob carga, {un}": fmt_lim(sol_n["final"].subs(valores),
                                                                   controle_tensao)})
    f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title=rot_resp)
    st.plotly_chart(f, width="stretch")
    if linhas:
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.caption("Maxwell responde instantaneamente pela mola, mas flui sem limite sob tensão constante "
               "e relaxa a tensão até zero. Kelvin-Voigt não tem resposta instantânea e tende a um "
               "valor de equilíbrio. O Sólido Linear Padrão combina os dois comportamentos: resposta "
               "instantânea e valor final limitado.")


# ---------------------------------------------------------------- aba: solução analítica
def passo(n, texto, *equacoes):
    st.markdown(f"**Passo {n}.** {texto}")
    for eq in equacoes:
        st.latex(eq)


def substituicao(n, *exprs):
    exibir = {m.E: E_, m.eta: eta_, m.tau: tau}
    if nome == "Sólido Linear Padrão":
        exibir[m.Einf] = Einf_
    exibir.update({k: v for k, v in valores.items() if k in (m.sig0, m.eps0, m.t1)})
    if omega is not None:
        exibir[m.w] = omega
    nums = ", \\quad ".join(f"{m.tex(k)} = {float(v):.4g}" for k, v in exibir.items())
    passo(n, "Substituição numérica dos parâmetros informados (tensões em MPa, deformações em mm/mm, "
             "tempo em s):", nums, *exprs)


with tabs[3]:
    P, Q = m.operadores(nome)
    passo(1, f"Equação constitutiva do modelo de {nome} na forma operacional "
             r"$P(D)\,\sigma = Q(D)\,\varepsilon$, com $D = d/dt$ (o ponto indica derivada no tempo) "
             r"e $\tau = \eta/E$:",
          m.equacao_diferencial(nome),
          rf"P(s) = {m.tex(P)}, \qquad Q(s) = {m.tex(Q)}")

    if ensaio != "senoidal":
        tipo = "relaxacao" if ensaio == "relaxacao" else "fluencia"
        sol = m.solucao_degrau(nome, tipo)
        if tipo == "fluencia":
            ent, sai, lado, amp = r"\sigma(t) = \sigma_0\,H(t)", r"\bar{\varepsilon}(s)", r"\varepsilon(t)", m.sig0
            fun, fun_nome, ent_s = "J", "fluência", r"\bar{\sigma}(s)"
            razao = r"\frac{P(s)}{Q(s)}\,\bar{\sigma}(s)"
        else:
            ent, sai, lado, amp = r"\varepsilon(t) = \varepsilon_0\,H(t)", r"\bar{\sigma}(s)", r"\sigma(t)", m.eps0
            fun, fun_nome, ent_s = "G", "relaxação", r"\bar{\varepsilon}(s)"
            razao = r"\frac{Q(s)}{P(s)}\,\bar{\varepsilon}(s)"
        texto2 = ("Carregamento aplicado (degrau, H é a função de Heaviside) e sua transformada de Laplace:"
                  if ensaio != "personalizado" else
                  "Para um histórico qualquer, primeiro obtém-se a função de fluência J(t), que é a resposta "
                  "a um degrau de tensão (H é a função de Heaviside):")
        passo(2, texto2, ent + r"\quad \Longrightarrow \quad " + ent_s + " = " + m.tex(sol["entrada_s"]))
        passo(3, "Com condições iniciais nulas (t = 0⁻), a equação diferencial vira uma equação algébrica em s:",
              rf"{sai} = {razao} = " + m.tex(sol["saida_s"]))
        passo(4, "Decomposição em frações parciais:", rf"{sai} = " + m.tex(sol["fracoes"]))
        passo(5, "Transformada inversa de Laplace, válida para t > 0:", rf"{lado} = " + m.tex(sol["saida_t"]))
        passo(6, f"Função de {fun_nome} (resposta por unidade de carregamento):",
              rf"{fun}(t) = \frac{{{lado}}}{{{m.tex(amp)}}} = " + m.tex(sp.expand(sol["saida_t"] / amp)))
        if nome == "Kelvin-Voigt" and tipo == "relaxacao":
            st.markdown(r"Observação: a transformada inversa do termo constante $E\,\tau\,\varepsilon_0$ é "
                        r"o impulso $\eta\,\varepsilon_0\,\delta(t)$, que atua apenas em $t = 0$.")
        passo(7, "Verificação física pelos limites. Em t → 0⁺ os amortecedores ainda não se deformaram e só "
                 "as molas respondem; em t → ∞ as taxas se anulam e os amortecedores deixam de transmitir "
                 "esforço:",
              rf"\lim_{{t \to 0^+}} {lado} = {m.tex(sol['inicial'])}, \qquad "
              rf"\lim_{{t \to \infty}} {lado} = {m.tex(sol['final'])}")

        if ensaio == "recuperacao":
            Jt = sol["saida_t"] / m.sig0
            descarga = sp.expand(m.sig0 * (Jt - Jt.subs(m.t, m.t - m.t1)))
            passo(8, "Forma integral (hereditária) e superposição de Boltzmann. A carga é "
                     r"$\sigma(t) = \sigma_0\,[H(t) - H(t - t_1)]$, logo:",
                  r"\varepsilon(t) = \int_{0^-}^{t} J(t-\xi)\,\frac{d\sigma(\xi)}{d\xi}\,d\xi"
                  r" = \sigma_0\,\big[J(t) - J(t - t_1)\,H(t - t_1)\big]",
                  r"\varepsilon(t) = " + m.tex(sol["saida_t"]) + r", \qquad 0 < t < t_1",
                  r"\varepsilon(t) = " + m.tex(descarga) + r", \qquad t \geq t_1")
            substituicao(9, r"\varepsilon(t) = " + m.tex(num(sol["saida_t"], valores)) + r", \qquad 0 < t < t_1",
                         r"\varepsilon(t) = " + m.tex(num(descarga, valores)) + r", \qquad t \geq t_1")
        elif ensaio == "personalizado":
            Jt = sp.expand(sol["saida_t"] / m.sig0)
            passo(8, "Forma integral (hereditária). Pelo princípio de superposição de Boltzmann, cada "
                     "incremento de tensão dσ(ξ) produz a resposta J(t − ξ) dσ(ξ). Para o histórico da "
                     "tabela, a integral é avaliada numericamente como uma soma sobre a malha de tempo:",
                  r"\varepsilon(t) = \int_{0^-}^{t} J(t-\xi)\,\frac{d\sigma(\xi)}{d\xi}\,d\xi"
                  r"\;\approx\; \sum_{\xi_j \le t} J(t-\xi_j)\,\Delta\sigma_j")
            substituicao(9, r"J(t) = " + m.tex(num(Jt, valores)) + r"\ \text{MPa}^{-1}")
        else:
            integral = (r"\varepsilon(t) = \int_{0^-}^{t} J(t-\xi)\,\frac{d\sigma(\xi)}{d\xi}\,d\xi"
                        if tipo == "fluencia" else
                        r"\sigma(t) = \int_{0^-}^{t} G(t-\xi)\,\frac{d\varepsilon(\xi)}{d\xi}\,d\xi")
            passo(8, f"Forma integral equivalente (válida para qualquer histórico de carregamento), "
                     f"usando a função de {fun_nome}:", integral)
            substituicao(9, rf"{lado} = " + m.tex(num(sol["saida_t"], valores)))

    else:
        sol = m.solucao_senoidal(nome)
        passo(2, "Deformação senoidal imposta a partir de t = 0 e sua transformada de Laplace:",
              r"\varepsilon(t) = \varepsilon_0 \sin(\omega t) \quad \Longrightarrow \quad "
              r"\bar{\varepsilon}(s) = " + m.tex(sol["entrada_s"]))
        passo(3, "Com condições iniciais nulas, a tensão no domínio de Laplace é:",
              r"\bar{\sigma}(s) = \frac{Q(s)}{P(s)}\,\bar{\varepsilon}(s) = " + m.tex(sol["saida_s"]))
        passo(4, "Decomposição em frações parciais:", r"\bar{\sigma}(s) = " + m.tex(sol["fracoes"]))
        passo(5, "Transformada inversa: solução completa, com o transitório (termos em e^(−t/τ)) e o "
                 "regime permanente (termos em seno e cosseno):",
              r"\sigma(t) = " + m.tex(sol["saida_t"]))
        passo(6, r"Para $t \gg \tau$ o transitório desaparece e a tensão fica harmônica. O regime permanente "
                 r"é obtido diretamente substituindo $D \to i\omega$, o que define o módulo complexo:",
              r"E^*(\omega) = \frac{Q(i\omega)}{P(i\omega)} = " + m.tex(sol["Ec"]),
              r"\sigma(t) = \varepsilon_0\left[E'\sin(\omega t) + E''\cos(\omega t)\right]"
              r" = \sigma_0 \sin(\omega t + \delta), \qquad \sigma_0 = \varepsilon_0\,|E^*|")
        passo(7, "Separando as partes real e imaginária:",
              r"E'(\omega) = \operatorname{Re}[E^*] = " + m.tex(sol["E_arm"]),
              r"E''(\omega) = \operatorname{Im}[E^*] = " + m.tex(sol["E_perda"]),
              r"\tan\delta = \frac{E''}{E'} = " + m.tex(sol["tan_d"]))
        passo(8, "Energia dissipada por ciclo (área do laço de histerese):",
              r"W_d = \oint \sigma\, d\varepsilon = \pi\, E''\, \varepsilon_0^2")
        v = {**valores, m.w: omega}
        Ea_, Ep_ = float(num(sol["E_arm"], v)), float(num(sol["E_perda"], v))
        e0 = float(valores[m.eps0])
        substituicao(9, r"\sigma(t) = " + m.tex(num(sol["saida_t"], v)) + r"\ \text{MPa}",
                     rf"E' = {Ea_:.4g}\ \text{{MPa}}, \quad E'' = {Ep_:.4g}\ \text{{MPa}}, \quad "
                     rf"\tan\delta = {Ep_ / Ea_:.4g}, \quad \delta = {np.degrees(np.arctan2(Ep_, Ea_)):.4g}^\circ",
                     rf"W_d = {np.pi * Ep_ * e0 ** 2:.4g}\ \text{{MJ/m}}^3")

# ---------------------------------------------------------------- aba: DMTA
if ensaio == "senoidal":
    with tabs[4]:
        st.markdown("A curva começa com um transitório. As grandezas dinâmicas se referem ao regime "
                    "permanente, por isso o laço de histerese usa apenas o último ciclo.")
        a, b = st.columns(2, gap="large")
        with a:
            e0 = float(valores[m.eps0])
            s0 = np.max(np.abs(sig)) or 1.0
            f = grafico(title="Deformação e tensão normalizadas")
            f.add_trace(go.Scatter(x=tt, y=eps / e0, name="ε / ε0", line=dict(color=COR_AMORT)))
            f.add_trace(go.Scatter(x=tt, y=sig / s0, name="σ / σmáx", line=dict(color=COR_MOLA)))
            f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title="Amplitude normalizada")
            st.plotly_chart(f, width="stretch")
        with b:
            ultimo = tt >= t_final - 1 / freq
            f = grafico(title="Laço de histerese no último ciclo (σ × ε)")
            f.add_trace(go.Scatter(x=eps[ultimo] * 100, y=sig[ultimo], line=dict(color=COR_MOLA),
                                   showlegend=False, fill="toself", fillcolor="rgba(31,78,121,0.08)"))
            f.update_layout(xaxis_title=ROT_EPS, yaxis_title=ROT_SIG)
            st.plotly_chart(f, width="stretch")

        ww = np.logspace(-2, 2, 300) / tau
        sol = m.solucao_senoidal(nome)
        Ea_w = m.numerica(sol["E_arm"], valores, m.w)(ww)
        Ep_w = m.numerica(sol["E_perda"], valores, m.w)(ww)
        fw = ww / (2 * np.pi)
        f = grafico(height=420, title="Varredura em frequência")
        f.add_trace(go.Scatter(x=fw, y=Ea_w, name="E′ armazenamento (MPa)", line=dict(color=COR_MOLA)))
        f.add_trace(go.Scatter(x=fw, y=Ep_w, name="E″ perda (MPa)", line=dict(color=COR_AMORT)))
        f.add_trace(go.Scatter(x=fw, y=Ep_w / Ea_w, name="tan δ", yaxis="y2",
                               line=dict(color="#4F8A5B", dash="dash")))
        f.add_vline(x=freq, line=dict(color="#5F6B7A", dash="dot"),
                    annotation_text="f atual", annotation_position="top")
        f.update_layout(xaxis=dict(type="log", dtick=1, title="Frequência, f (Hz)", **GRADE),
                        yaxis=dict(title="Módulo (MPa)", rangemode="tozero", **GRADE),
                        yaxis2=dict(title="tan δ", rangemode="tozero", tickformat=".2f", overlaying="y",
                                    side="right", showgrid=False))
        st.plotly_chart(f, width="stretch")
        st.caption("E′ mede a energia armazenada e recuperada por ciclo (resposta elástica); "
                   "E″ mede a energia dissipada (resposta viscosa); tan δ = E″/E′ é o fator de "
                   "amortecimento. Em ensaios de DMTA esses parâmetros são medidos em função da "
                   "frequência e da temperatura.")
