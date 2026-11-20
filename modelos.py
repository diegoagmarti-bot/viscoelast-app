"""Modelos reológicos da viscoelasticidade linear unidimensional.

Cada modelo é descrito pela forma operacional da equação constitutiva:

    p0*sigma + p1*dsigma/dt = q0*epsilon + q1*depsilon/dt

Os três modelos usam a mesma mola E e o mesmo amortecedor eta, com
tau = eta/E. O Sólido Linear Padrão acrescenta a mola isolada E_inf.

As soluções analíticas são obtidas com sympy pela transformada de Laplace.
"""
from functools import lru_cache

import numpy as np
import sympy as sp

# Símbolos
t, s, w, tau = sp.symbols("t s omega tau", positive=True)
t1 = sp.Symbol("t_1", positive=True)
sig0 = sp.Symbol("sigma_0", positive=True)
eps0 = sp.Symbol("epsilon_0", positive=True)
E, eta, Einf = sp.symbols("E eta E_infty", positive=True)

NOMES_LATEX = {Einf: r"E_{\infty}"}


def tex(expr):
    """LaTeX de uma expressão sympy."""
    return sp.latex(expr, symbol_names=NOMES_LATEX)


MODELOS = {
    "Maxwell": {
        "arranjo": "Mola E e amortecedor η em série",
        "P": (1, tau),
        "Q": (0, E * tau),
        "tau_nome": "Tempo de relaxação τ = η/E (s)",
    },
    "Kelvin-Voigt": {
        "arranjo": "Mola E e amortecedor η em paralelo",
        "P": (1, 0),
        "Q": (E, E * tau),
        "tau_nome": "Tempo de retardo τ = η/E (s)",
    },
    "Sólido Linear Padrão": {
        "arranjo": "Mola E∞ em paralelo com um braço de Maxwell (mola E + amortecedor η)",
        "P": (1, tau),
        "Q": (Einf, (Einf + E) * tau),
        "tau_nome": "Tempo de relaxação τ = η/E (s)",
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


def _inversa(saida_s):
    fracoes = sp.apart(sp.cancel(saida_s), s)
    saida_t = sp.inverse_laplace_transform(fracoes, s, t).subs(sp.Heaviside(t), 1)
    return fracoes, sp.expand(saida_t)


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
    fracoes, saida_t = _inversa(saida_s)
    return {"P": P, "Q": Q, "entrada_s": entrada, "saida_s": saida_s,
            "fracoes": fracoes, "saida_t": saida_t,
            "inicial": sp.limit(saida_t, t, 0, "+"), "final": sp.limit(saida_t, t, sp.oo)}


@lru_cache(maxsize=None)
def solucao_senoidal(nome):
    """Solução completa para epsilon = eps0*sin(w t) e grandezas dinâmicas."""
    P, Q = operadores(nome)
    entrada = eps0 * w / (s**2 + w**2)
    saida_s = entrada * Q / P
    fracoes, saida_t = _inversa(saida_s)
    Ec = sp.simplify(Q.subs(s, sp.I * w) / P.subs(s, sp.I * w))
    E_arm = sp.simplify(sp.re(Ec))
    E_perda = sp.simplify(sp.im(Ec))
    return {"entrada_s": entrada, "saida_s": saida_s, "fracoes": fracoes, "saida_t": saida_t,
            "Ec": Ec, "E_arm": E_arm, "E_perda": E_perda,
            "tan_d": sp.simplify(E_perda / E_arm)}


def numerica(expr, valores, var=t):
    """Converte uma expressão em função numpy vetorizada (impulsos de Dirac são descartados)."""
    expr = expr.subs(valores).replace(sp.DiracDelta, lambda *a: 0)
    f = sp.lambdify(var, expr, "numpy")
    return lambda x: np.asarray(f(x), dtype=float) * np.ones_like(x)


def boltzmann(J, tt, sig):
    """Integral hereditária eps(t) = soma de J(t - xi) * d(sigma)(xi) em uma malha uniforme."""
    dsig = np.diff(sig, prepend=0.0)
    atraso = tt[:, None] - tt[None, :]
    ativo = atraso >= 0
    return (J(np.where(ativo, atraso, 0.0)) * ativo) @ dsig


def simular(nome, ensaio, valores, tt, t_rem=None, omega=None, historico=None):
    """Históricos de tensão e deformação (sigma, epsilon) nos instantes tt."""
    if ensaio == "senoidal":
        sig = numerica(solucao_senoidal(nome)["saida_t"].subs(w, omega), valores)(tt)
        return sig, float(valores[eps0]) * np.sin(omega * tt)

    if ensaio == "relaxacao":
        sig = numerica(solucao_degrau(nome, "relaxacao")["saida_t"], valores)(tt)
        return sig, np.full_like(tt, float(valores[eps0]))

    J = numerica(sp.expand(solucao_degrau(nome, "fluencia")["saida_t"] / sig0), valores)
    if ensaio == "personalizado":
        tempos, tensoes = historico
        sig = np.interp(tt, tempos, tensoes, left=0.0)
        return sig, boltzmann(J, tt, sig)

    s0 = float(valores[sig0])
    if ensaio == "fluencia":
        return np.full_like(tt, s0), s0 * J(tt)

    # Fluência e recuperação: superposição de Boltzmann de dois degraus
    descarregado = tt >= t_rem
    eps = s0 * (J(tt) - np.where(descarregado, J(np.maximum(tt - t_rem, 0.0)), 0.0))
    return np.where(descarregado, 0.0, s0), eps


def componentes(nome, sig, eps, valores):
    """Deformação e tensão em cada mola e amortecedor."""
    Ev = float(valores[E])
    if nome == "Maxwell":
        e_mola = sig / Ev
        return {"Mola E": (e_mola, sig), "Amortecedor η": (eps - e_mola, sig)}
    if nome == "Kelvin-Voigt":
        s_mola = Ev * eps
        return {"Mola E": (eps, s_mola), "Amortecedor η": (eps, sig - s_mola)}
    s_inf = float(valores[Einf]) * eps
    s_braco = sig - s_inf
    e_mola = s_braco / Ev
    return {"Mola E∞": (eps, s_inf), "Mola E": (e_mola, s_braco),
            "Amortecedor η": (eps - e_mola, s_braco)}
