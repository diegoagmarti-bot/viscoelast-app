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


def arredondar(expr, digitos=4):
    """Avalia numericamente e arredonda todos os números para exibição."""
    expr = sp.N(expr, 15)
    return expr.xreplace({x: sp.Float(float(f"{float(x):.{digitos}g}"), digitos) for x in expr.atoms(sp.Float)})


def numerica(expr, valores, var=t):
    """Converte uma expressão em função numpy vetorizada (impulsos de Dirac são descartados)."""
    expr = expr.subs(valores).replace(sp.DiracDelta, lambda *a: 0)
    f = sp.lambdify(var, expr, "numpy")
    return lambda x: np.asarray(f(x), dtype=float) * np.ones_like(x)


@lru_cache(maxsize=None)
def funcoes_hereditarias(nome, controle):
    """Função de fluência J(t) ('tensao') ou de relaxação G(t) ('deformacao') e sua integral.

    A integral F_int(t) = integral de 0 a t de F(u) du é a resposta a uma rampa de taxa unitária.
    No Kelvin-Voigt, G(t) tem o impulso eta*delta(t), que contribui com a constante eta em F_int.
    """
    u = sp.Symbol("u", positive=True)
    if controle == "tensao":
        F = sp.expand(solucao_degrau(nome, "fluencia")["saida_t"] / sig0)
        impulso = 0
    else:
        F = sp.expand(solucao_degrau(nome, "relaxacao")["saida_t"] / eps0)
        (_, p1), (_, q1) = MODELOS[nome]["P"], MODELOS[nome]["Q"]
        impulso = q1 if p1 == 0 else 0
    F_int = sp.expand(sp.integrate(F.subs(t, u), (u, 0, t)) + impulso)
    return F, F_int


def trechos(historico):
    """Converte pontos (tempo, valor) ligados por retas em degraus e rampas.

    Retorna (degraus, rampas): degraus = [(a, salto)], rampas = [(a, b, taxa)].
    Antes do primeiro ponto o valor é zero; depois do último, constante.
    """
    tempos, valores = historico
    degraus, rampas = [], []
    if valores[0] != 0:
        degraus.append((tempos[0], valores[0]))
    for (ta, va), (tb, vb) in zip(zip(tempos, valores), zip(tempos[1:], valores[1:])):
        if tb == ta:
            if vb != va:
                degraus.append((ta, vb - va))
        elif vb != va:
            rampas.append((ta, tb, (vb - va) / (tb - ta)))
    return degraus, rampas


def resposta_historico(F, F_int, historico, tt):
    """Superposição de Boltzmann exata para um histórico linear por trechos (F e F_int numéricas)."""
    degraus, rampas = trechos(historico)
    saida = np.zeros_like(tt)

    def atrasada(f, a):
        return np.where(tt >= a, f(np.maximum(tt - a, 0.0)), 0.0)

    for a, salto in degraus:
        saida += salto * atrasada(F, a)
    for a, b, taxa in rampas:
        saida += taxa * (atrasada(F_int, a) - atrasada(F_int, b))
    return saida


def expressoes_por_intervalo(F, F_int, historico, valores, max_intervalos=6):
    """Expressões fechadas da resposta em cada intervalo entre pontos do histórico."""
    degraus, rampas = trechos(historico)
    cortes = sorted(set(historico[0]))
    Fn, Fin = F.subs(valores), F_int.subs(valores)
    saida = []
    for c0, c1 in zip(cortes, cortes[1:] + [sp.oo]):
        expr = sum(salto * Fn.subs(t, t - a) for a, salto in degraus if a <= c0)
        for a, b, taxa in rampas:
            if a <= c0:
                expr += taxa * Fin.subs(t, t - a)
            if b <= c0:
                expr -= taxa * Fin.subs(t, t - b)
        expr = arredondar(sp.expand(sp.sympify(expr)))
        if saida and saida[-1][2] == expr:  # mesmo trecho: estende o intervalo anterior
            saida[-1] = (saida[-1][0], c1, expr)
        else:
            saida.append((c0, c1, expr))
    return saida[:max_intervalos]


def simular(nome, ensaio, valores, tt, t_rem=None, omega=None, historico=None):
    """Históricos de tensão e deformação (sigma, epsilon) nos instantes tt."""
    if ensaio == "senoidal":
        sig = numerica(solucao_senoidal(nome)["saida_t"].subs(w, omega), valores)(tt)
        return sig, float(valores[eps0]) * np.sin(omega * tt)

    if ensaio == "relaxacao":
        sig = numerica(solucao_degrau(nome, "relaxacao")["saida_t"], valores)(tt)
        return sig, np.full_like(tt, float(valores[eps0]))

    if ensaio in ("hist_tensao", "hist_deformacao"):
        controle = "tensao" if ensaio == "hist_tensao" else "deformacao"
        F, F_int = funcoes_hereditarias(nome, controle)
        entrada = np.interp(tt, *historico, left=0.0)
        saida = resposta_historico(numerica(F, valores), numerica(F_int, valores), historico, tt)
        return (entrada, saida) if controle == "tensao" else (saida, entrada)

    J = numerica(sp.expand(solucao_degrau(nome, "fluencia")["saida_t"] / sig0), valores)
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


def energias(comp):
    """Energia por unidade de volume (MJ/m3 = MPa): armazenada nas molas e dissipada no amortecedor."""
    saida = {}
    for rot, (e_c, s_c) in comp.items():
        if rot.startswith("Mola"):
            saida[f"Armazenada, {rot}"] = 0.5 * s_c * e_c
        else:
            trabalho = 0.5 * (s_c[1:] + s_c[:-1]) * np.diff(e_c)
            saida[f"Dissipada, {rot}"] = np.concatenate([[0.0], np.cumsum(trabalho)])
    return saida
