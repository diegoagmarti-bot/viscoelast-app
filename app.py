"""Viscoelasticidade linear unidimensional: aplicativo Streamlit."""
import numpy as np
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
    "Carga senoidal (DMTA)": "senoidal",
}
CORES = [COR_MOLA, COR_AMORT, "#4F8A5B", "#7A5C99"]
GRADE = dict(showgrid=True, gridcolor="#E6E9EE")


def grafico(height=360, **layout):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=20, t=80, b=50),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **layout)
    fig.update_xaxes(**GRADE)
    fig.update_yaxes(**GRADE)
    return fig


def num(expr, valores):
    """Substitui os valores numéricos e arredonda para exibição."""
    return sp.N(expr.subs(valores), 4)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuração")
    nome = st.selectbox("Modelo reológico", list(m.MODELOS))
    ensaio_rot = st.selectbox("Histórico de carregamento", list(ENSAIOS))
    ensaio = ENSAIOS[ensaio_rot]
    modelo = m.MODELOS[nome]

    st.subheader("Parâmetros do material")
    valores = {}
    for simb, rotulo, padrao in modelo["params"]:
        valores[simb] = st.number_input(rotulo, min_value=1e-6, value=padrao, format="%.4g")
    tau = float(modelo["tau"].subs(valores))

    st.subheader("Carregamento")
    t_rem = omega = None
    if ensaio in ("fluencia", "recuperacao"):
        valores[m.sig0] = st.number_input("Tensão aplicada, σ0 (MPa)", min_value=1e-6, value=10.0,
                                          format="%.4g")
    else:
        valores[m.eps0] = st.number_input("Amplitude de deformação, ε0 (mm/mm)", min_value=1e-9,
                                          value=0.005, format="%.4g")
    if ensaio == "senoidal":
        freq = st.number_input("Frequência, f (Hz)", min_value=1e-6, value=round(1 / (2 * np.pi * tau), 4),
                               format="%.4g", help="Valor inicial: ω = 1/τ, sendo τ o tempo característico do modelo.")
        ciclos = st.slider("Número de ciclos exibidos", 1, 10, 3)
        omega = 2 * np.pi * freq
        t_final = ciclos / freq
    else:
        t_final = st.number_input("Tempo total de análise (s)", min_value=1e-6, value=round(6 * tau, 4),
                                  format="%.4g")
    if ensaio == "recuperacao":
        t_rem = st.number_input("Instante de remoção da carga, t1 (s)", min_value=1e-6,
                                max_value=float(t_final), value=float(t_final) / 2, format="%.4g")
        valores[m.t1] = t_rem

# ---------------------------------------------------------------- simulação
tt = np.linspace(0, t_final, 1200)
sig, eps = m.simular(nome, ensaio, valores, tt, t_rem=t_rem, omega=omega)
comp = m.componentes(nome, sig, eps, valores)
controle_tensao = ensaio in ("fluencia", "recuperacao")
if controle_tensao:
    resposta, rot_resp = eps, "Deformação, ε (mm/mm)"
else:
    resposta, rot_resp = sig, "Tensão, σ (MPa)"

# ---------------------------------------------------------------- cabeçalho
st.title("Viscoelasticidade Linear Unidimensional")
st.caption(f"Modelo de {nome}  ·  {modelo['arranjo']}  ·  {ensaio_rot}")

st.markdown("**Equação constitutiva diferencial**")
st.latex(m.equacao_diferencial(nome))
c1, c2, c3, c4 = st.columns(4)
if ensaio == "senoidal":
    sol_s = m.solucao_senoidal(nome)
    Ea = float(sol_s["E_arm"].subs(valores).subs(m.w, omega))
    Ep = float(sol_s["E_perda"].subs(valores).subs(m.w, omega))
    c1.metric("E′, armazenamento (MPa)", f"{Ea:.4g}", border=True)
    c2.metric("E″, perda (MPa)", f"{Ep:.4g}", border=True)
    c3.metric("tan δ", f"{Ep / Ea:.4g}", border=True)
    c4.metric("Ângulo de fase δ (graus)", f"{np.degrees(np.arctan2(Ep, Ea)):.4g}", border=True)
else:
    tipo = "relaxacao" if ensaio == "relaxacao" else "fluencia"
    expr = m.solucao_degrau(nome, tipo)["saida_t"].subs(valores)
    if ensaio == "recuperacao":  # deformação residual após a remoção da carga
        lim = sp.limit(expr - expr.subs(m.t, m.t - t_rem), m.t, sp.oo)
    else:
        lim = sp.limit(expr, m.t, sp.oo)
    rot = "σ" if tipo == "relaxacao" else "ε"
    inicial = float(expr.replace(sp.DiracDelta, lambda *a: 0).subs(m.t, 0))
    c1.metric(modelo["tau_nome"], f"{tau:.4g}", border=True)
    c2.metric(f"Resposta instantânea {rot}(0⁺)", f"{inicial:.4g}", border=True)
    c3.metric(f"{rot} no fim da análise", f"{resposta[-1]:.4g}", border=True)
    c4.metric(f"{rot}(t → ∞)" + (" residual" if ensaio == "recuperacao" else ""),
              "ilimitada" if lim.is_infinite else f"{float(lim):.4g}", border=True)

abas = [":material/animation: Simulação", ":material/stacked_line_chart: Componentes",
        ":material/function: Solução analítica"]
if ensaio == "senoidal":
    abas.append(":material/graphic_eq: Análise dinâmica (DMTA)")
tabs = st.tabs(abas)

# ---------------------------------------------------------------- aba: simulação
with tabs[0]:
    st.markdown("Use **Reproduzir** ou arraste o controle de tempo para ver molas e amortecedores "
                "se deformarem. O comprimento de cada elemento é proporcional à sua deformação.")
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
        f.add_trace(go.Scatter(x=tt, y=eps, name="Deformação ε", yaxis="y2", line=dict(color=COR_AMORT)))
        f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title="Tensão, σ (MPa)",
                        yaxis2=dict(title="Deformação, ε (mm/mm)", overlaying="y", side="right",
                                    showgrid=False))
        st.plotly_chart(f, width="stretch")
    with b:
        f = grafico(title="Deformação em cada componente")
        for (rot, (e_c, _)), cor in zip(comp.items(), CORES):
            f.add_trace(go.Scatter(x=tt, y=e_c, name=rot, line=dict(color=cor)))
        f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title="Deformação (mm/mm)")
        st.plotly_chart(f, width="stretch")
    f = grafico(title="Tensão em cada componente")
    for (rot, (_, s_c)), cor in zip(comp.items(), CORES):
        f.add_trace(go.Scatter(x=tt, y=s_c, name=rot, line=dict(color=cor)))
    f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title="Tensão (MPa)")
    st.plotly_chart(f, width="stretch")
    st.caption("A mola responde instantaneamente (σ = E ε). O amortecedor responde à taxa de "
               "deformação (σ = η dε/dt). Em série, os elementos compartilham a tensão e somam as "
               "deformações; em paralelo, compartilham a deformação e somam as tensões.")


# ---------------------------------------------------------------- aba: solução analítica
def passo(n, texto, *equacoes):
    st.markdown(f"**Passo {n}.** {texto}")
    for eq in equacoes:
        st.latex(eq)


with tabs[2]:
    P, Q = m.operadores(nome)
    passo(1, f"Equação constitutiva do modelo de {nome} na forma operacional "
             r"$P(D)\,\sigma = Q(D)\,\varepsilon$, com $D = d/dt$ (o ponto indica derivada no tempo):",
          m.equacao_diferencial(nome),
          rf"P(s) = {m.tex(P)}, \qquad Q(s) = {m.tex(Q)}")

    if ensaio in ("fluencia", "relaxacao", "recuperacao"):
        tipo = "relaxacao" if ensaio == "relaxacao" else "fluencia"
        sol = m.solucao_degrau(nome, tipo)
        if tipo == "fluencia":
            ent, sai, fun, fun_nome = r"\sigma(t) = \sigma_0\,H(t)", r"\bar{\varepsilon}(s)", "J", "fluência"
        else:
            ent, sai, fun, fun_nome = r"\varepsilon(t) = \varepsilon_0\,H(t)", r"\bar{\sigma}(s)", "G", "relaxação"
        lado = r"\varepsilon(t)" if tipo == "fluencia" else r"\sigma(t)"
        amp = m.sig0 if tipo == "fluencia" else m.eps0

        passo(2, "Carregamento aplicado (degrau, H é a função de Heaviside) e sua transformada de Laplace:",
              ent + r"\quad \Longrightarrow \quad " +
              (r"\bar{\sigma}(s) = " if tipo == "fluencia" else r"\bar{\varepsilon}(s) = ") + m.tex(sol["entrada_s"]))
        passo(3, "Com condições iniciais nulas (t = 0⁻), a equação diferencial vira uma equação algébrica em s:",
              (rf"{sai} = \frac{{P(s)}}{{Q(s)}}\,\bar{{\sigma}}(s) = " if tipo == "fluencia"
               else rf"{sai} = \frac{{Q(s)}}{{P(s)}}\,\bar{{\varepsilon}}(s) = ") + m.tex(sol["saida_s"]))
        passo(4, "Decomposição em frações parciais:", rf"{sai} = " + m.tex(sol["fracoes"]))
        passo(5, "Transformada inversa de Laplace, válida para t > 0:", rf"{lado} = " + m.tex(sol["saida_t"]))
        passo(6, f"Função de {fun_nome} (resposta por unidade de carregamento):",
              rf"{fun}(t) = \frac{{{lado}}}{{{m.tex(amp)}}} = " + m.tex(sp.expand(sol["saida_t"] / amp)))
        if nome == "Kelvin-Voigt" and tipo == "relaxacao":
            st.markdown(r"Observação: a transformada inversa do termo constante $\eta\,\varepsilon_0$ é "
                        r"o impulso $\eta\,\varepsilon_0\,\delta(t)$, que atua apenas em $t = 0$.")
        n = 7
        if ensaio == "recuperacao":
            Jt = sol["saida_t"] / m.sig0
            descarga = sp.expand(m.sig0 * (Jt - Jt.subs(m.t, m.t - m.t1)))
            passo(7, "Forma integral (hereditária) e superposição de Boltzmann. A carga é "
                     r"$\sigma(t) = \sigma_0\,[H(t) - H(t - t_1)]$, logo:",
                  r"\varepsilon(t) = \int_{0^-}^{t} J(t-\xi)\,\frac{d\sigma(\xi)}{d\xi}\,d\xi"
                  r" = \sigma_0\,\big[J(t) - J(t - t_1)\,H(t - t_1)\big]",
                  r"\varepsilon(t) = " + m.tex(sol["saida_t"]) + r", \qquad 0 < t < t_1",
                  r"\varepsilon(t) = " + m.tex(descarga) + r", \qquad t \geq t_1")
            n = 8
        else:
            integral = (r"\varepsilon(t) = \int_{0^-}^{t} J(t-\xi)\,\frac{d\sigma(\xi)}{d\xi}\,d\xi"
                        if tipo == "fluencia" else
                        r"\sigma(t) = \int_{0^-}^{t} G(t-\xi)\,\frac{d\varepsilon(\xi)}{d\xi}\,d\xi")
            passo(7, f"Forma integral equivalente (válida para qualquer histórico de carregamento), "
                     f"usando a função de {fun_nome}:", integral)
            n = 8
        nums = ", \\quad ".join(f"{m.tex(k)} = {float(v):.4g}" for k, v in valores.items())
        exprs = [rf"{lado} = " + m.tex(num(sol["saida_t"], valores))]
        if ensaio == "recuperacao":
            exprs[0] += r", \qquad 0 < t < t_1"
            exprs.append(r"\varepsilon(t) = " + m.tex(num(descarga, valores)) + r", \qquad t \geq t_1")
        passo(n, "Substituição numérica dos parâmetros informados:", nums, *exprs)

    else:
        sol = m.solucao_senoidal(nome)
        passo(2, "Deformação senoidal imposta, escrita na forma complexa:",
              r"\varepsilon(t) = \varepsilon_0 \sin(\omega t) = \operatorname{Im}\left[\varepsilon_0\, e^{i\omega t}\right]")
        passo(3, r"Em regime permanente a tensão também é harmônica, $\sigma = \operatorname{Im}[\sigma^* e^{i\omega t}]$. "
                 r"Substituindo $D \to i\omega$ na equação constitutiva, obtém-se o módulo complexo:",
              r"E^*(\omega) = \frac{Q(i\omega)}{P(i\omega)} = " + m.tex(sol["Ec"]))
        passo(4, "Separando as partes real e imaginária:",
              r"E'(\omega) = \operatorname{Re}[E^*] = " + m.tex(sol["E_arm"]),
              r"E''(\omega) = \operatorname{Im}[E^*] = " + m.tex(sol["E_perda"]),
              r"\tan\delta = \frac{E''}{E'} = " + m.tex(sol["tan_d"]))
        passo(5, "Resposta em tensão, defasada de δ em relação à deformação:",
              r"\sigma(t) = \varepsilon_0\left[E'\sin(\omega t) + E''\cos(\omega t)\right]"
              r" = \sigma_0 \sin(\omega t + \delta), \qquad \sigma_0 = \varepsilon_0\,|E^*|")
        passo(6, "Energia dissipada por ciclo (área do laço de histerese):",
              r"W_d = \oint \sigma\, d\varepsilon = \pi\, E''\, \varepsilon_0^2")
        v = {**valores, m.w: omega}
        Ea_, Ep_ = float(num(sol["E_arm"], v)), float(num(sol["E_perda"], v))
        nums = ", \\quad ".join(f"{m.tex(k)} = {float(x):.4g}" for k, x in v.items())
        passo(7, "Substituição numérica dos parâmetros informados:", nums,
              rf"E' = {Ea_:.4g}\ \text{{MPa}}, \quad E'' = {Ep_:.4g}\ \text{{MPa}}, \quad "
              rf"\tan\delta = {Ep_ / Ea_:.4g}, \quad \delta = {np.degrees(np.arctan2(Ep_, Ea_)):.4g}^\circ",
              rf"\sigma(t) = {float(valores[m.eps0]) * Ea_:.4g}\,\sin({omega:.4g}\,t) + "
              rf"{float(valores[m.eps0]) * Ep_:.4g}\,\cos({omega:.4g}\,t)\ \text{{MPa}}",
              rf"W_d = {np.pi * Ep_ * float(valores[m.eps0]) ** 2:.4g}\ \text{{MJ/m}}^3")

# ---------------------------------------------------------------- aba: DMTA
if ensaio == "senoidal":
    with tabs[3]:
        a, b = st.columns(2, gap="large")
        with a:
            e0 = float(valores[m.eps0])
            s0 = np.max(np.abs(sig)) or 1.0
            f = grafico(title="Deformação e tensão normalizadas")
            f.add_trace(go.Scatter(x=tt, y=eps / e0, name="ε / ε0", line=dict(color=COR_AMORT)))
            f.add_trace(go.Scatter(x=tt, y=sig / s0, name="σ / σ0", line=dict(color=COR_MOLA)))
            f.update_layout(xaxis_title="Tempo, t (s)", yaxis_title="Amplitude normalizada")
            st.plotly_chart(f, width="stretch")
        with b:
            f = grafico(title="Laço de histerese (σ × ε)")
            f.add_trace(go.Scatter(x=eps, y=sig, line=dict(color=COR_MOLA), showlegend=False,
                                   fill="toself", fillcolor="rgba(31,78,121,0.08)"))
            f.update_layout(xaxis_title="Deformação, ε (mm/mm)", yaxis_title="Tensão, σ (MPa)")
            st.plotly_chart(f, width="stretch")

        ww = np.logspace(-2, 2, 300) / tau
        sol = m.solucao_senoidal(nome)
        Ea_w = sp.lambdify(m.w, sol["E_arm"].subs(valores), "numpy")(ww) * np.ones_like(ww)
        Ep_w = sp.lambdify(m.w, sol["E_perda"].subs(valores), "numpy")(ww) * np.ones_like(ww)
        fw = ww / (2 * np.pi)
        f = grafico(height=420, title="Varredura em frequência")
        f.add_trace(go.Scatter(x=fw, y=Ea_w, name="E′ armazenamento (MPa)", line=dict(color=COR_MOLA)))
        f.add_trace(go.Scatter(x=fw, y=Ep_w, name="E″ perda (MPa)", line=dict(color=COR_AMORT)))
        f.add_trace(go.Scatter(x=fw, y=Ep_w / Ea_w, name="tan δ", yaxis="y2",
                               line=dict(color="#4F8A5B", dash="dash")))
        f.add_vline(x=omega / (2 * np.pi), line=dict(color="#5F6B7A", dash="dot"),
                    annotation_text="f atual", annotation_position="top")
        f.update_layout(xaxis=dict(type="log", dtick=1, title="Frequência, f (Hz)", **GRADE),
                        yaxis=dict(title="Módulo (MPa)", rangemode="tozero", **GRADE),
                        yaxis2=dict(title="tan δ", rangemode="tozero", tickformat=".2f", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(f, width="stretch")
        st.caption("E′ mede a energia armazenada e recuperada por ciclo (resposta elástica); "
                   "E″ mede a energia dissipada (resposta viscosa); tan δ = E″/E′ é o fator de "
                   "amortecimento. Em ensaios de DMTA esses parâmetros são medidos em função da "
                   "frequência e da temperatura.")
