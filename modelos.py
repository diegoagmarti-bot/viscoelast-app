"""Modelos reológicos da viscoelasticidade linear unidimensional.

Cada modelo é descrito pela forma operacional da equação constitutiva:

    p0*sigma + p1*dsigma/dt = q0*epsilon + q1*depsilon/dt

As soluções analíticas são obtidas com sympy pela transformada de Laplace
(ensaios de fluência e relaxação) e pelo módulo complexo (carga senoidal).
"""
from functools import lru_cache

import numpy as np
import sympy as sp

# Símbolos
t, s, w = sp.symbols("t s omega", positive=True)
t1 = sp.Symbol("t_1", positive=True)
sig0 = sp.Symbol("sigma_0", positive=True)
eps0 = sp.Symbol("epsilon_0", positive=True)
E, eta, Einf, E1 = sp.symbols("E eta E_infty E_1", positive=True)

NOMES_LATEX = {Einf: r"E_{\infty}"}


def tex(expr):
    """LaTeX de uma expressão sympy."""
    return sp.latex(expr, symbol_names=NOMES_LATEX)


MODELOS = {
    "Maxwell": {
        "arranjo": "Mola e amortecedor em série",
        "params": [
            (E, "Módulo da mola, E (MPa)", 2000.0),
            (eta, "Viscosidade do amortecedor, η (MPa·s)", 10000.0),
        ],
        "P": (1, eta / E),
        "Q": (0, eta),
        "tau": eta / E,
        "tau_nome": "Tempo de relaxação τ = η/E (s)",
    },
    "Kelvin-Voigt": {
        "arranjo": "Mola e amortecedor em paralelo",
        "params": [
            (E, "Módulo da mola, E (MPa)", 2000.0),
            (eta, "Viscosidade do amortecedor, η (MPa·s)", 10000.0),
        ],
        "P": (1, 0),
        "Q": (E, eta),
        "tau": eta / E,
        "tau_nome": "Tempo de retardo τ = η/E (s)",
    },
    "Sólido Linear Padrão": {
        "arranjo": "Mola E∞ em paralelo com um braço de Maxwell (mola E1 + amortecedor η)",
        "params": [
            (Einf, "Módulo da mola isolada, E∞ (MPa)", 1000.0),
            (E1, "Módulo da mola do braço de Maxwell, E1 (MPa)", 2000.0),
            (eta, "Viscosidade do amortecedor, η (MPa·s)", 10000.0),
        ],
        "P": (1, eta / E1),
        "Q": (Einf, eta * (Einf + E1) / E1),
        "tau": eta / E1,
        "tau_nome": "Tempo de relaxação τ = η/E1 (s)",
    },
}


def operadores(nome):
    """Polinômios P(s) e Q(s) da forma P*sigma = Q*epsilon."""
    (p0, p1), (q0, q1) = MODELOS[nome]["P"], MODELOS[nome]["Q"]
    return p0 + p1 * s, q0 + q1 * s


def equacao_diferencial(nome):
    """LaTeX da equação p0*sigma + p1*dsigma/dt = q0*epsilon + q1*depsilon/dt."""
    (p0, p1), (q0, q1) = MODELOS[nome]["P"], MODELOS[nome]["Q"]

    def lado(c0, c1, var):
        termos = []
        for c, v in ((c0, var), (c1, rf"\dot{{{var}}}")):
            if c == 0:
                continue
            termos.append(v if c == 1 else rf"{tex(c)}\,{v}")
        return " + ".join(termos)

    return lado(p0, p1, r"\sigma") + " = " + lado(q0, q1, r"\varepsilon")


@lru_cache(maxsize=None)
def solucao_degrau(nome, tipo):
    """Resposta a um degrau de tensão ('fluencia') ou de deformação ('relaxacao')."""
    P, Q = operadores(nome)
    if tipo == "fluencia":
        entrada = sig0 / s
        saida_s = entrada * P / Q
    else:
        entrada = eps0 / s
        saida_s = entrada * Q / P
    fracoes = sp.apart(sp.cancel(saida_s), s)
    saida_t = sp.inverse_laplace_transform(fracoes, s, t).subs(sp.Heaviside(t), 1)
    saida_t = sp.expand(saida_t)
    return {"P": P, "Q": Q, "entrada_s": entrada, "saida_s": saida_s,
            "fracoes": fracoes, "saida_t": saida_t}


@lru_cache(maxsize=None)
def solucao_senoidal(nome):
    """Módulo complexo, de armazenamento, de perda e tangente de delta."""
    P, Q = operadores(nome)
    Ec = sp.simplify(Q.subs(s, sp.I * w) / P.subs(s, sp.I * w))
    E_arm = sp.simplify(sp.re(Ec))
    E_perda = sp.simplify(sp.im(Ec))
    tan_d = sp.simplify(E_perda / E_arm)
    return {"Ec": Ec, "E_arm": E_arm, "E_perda": E_perda, "tan_d": tan_d}


def numerica(expr, valores):
    """Converte uma expressão em t em função numpy (vetorizada)."""
    f = sp.lambdify(t, expr.subs(valores), "numpy")
    return lambda tt: np.asarray(f(tt), dtype=float) * np.ones_like(tt)


def simular(nome, ensaio, valores, tt, t_rem=None, omega=None):
    """Históricos de tensão e deformação (sigma, epsilon) para os vetores de tempo tt."""
    if ensaio == "senoidal":
        sol = solucao_senoidal(nome)
        Ea = float(sol["E_arm"].subs(valores).subs(w, omega))
        Ep = float(sol["E_perda"].subs(valores).subs(w, omega))
        e0 = float(valores[eps0])
        return e0 * (Ea * np.sin(omega * tt) + Ep * np.cos(omega * tt)), e0 * np.sin(omega * tt)

    if ensaio == "relaxacao":
        sig = numerica(solucao_degrau(nome, "relaxacao")["saida_t"], valores)(tt)
        return sig, np.full_like(tt, float(valores[eps0]))

    eps_f = numerica(solucao_degrau(nome, "fluencia")["saida_t"], valores)
    s0 = float(valores[sig0])
    if ensaio == "fluencia":
        return np.full_like(tt, s0), eps_f(tt)

    # Fluência e recuperação: superposição de Boltzmann
    descarregado = tt >= t_rem
    eps = eps_f(tt) - np.where(descarregado, eps_f(np.maximum(tt - t_rem, 0.0)), 0.0)
    return np.where(descarregado, 0.0, s0), eps


def componentes(nome, sig, eps, valores):
    """Deformação e tensão em cada mola e amortecedor."""
    v = {k: float(x) for k, x in valores.items()}
    if nome == "Maxwell":
        e_mola = sig / v[E]
        return {"Mola E": (e_mola, sig), "Amortecedor η": (eps - e_mola, sig)}
    if nome == "Kelvin-Voigt":
        s_mola = v[E] * eps
        return {"Mola E": (eps, s_mola), "Amortecedor η": (eps, sig - s_mola)}
    s_inf = v[Einf] * eps
    s_braco = sig - s_inf
    e_mola1 = s_braco / v[E1]
    return {"Mola E∞": (eps, s_inf), "Mola E1": (e_mola1, s_braco),
            "Amortecedor η": (eps - e_mola1, s_braco)}
