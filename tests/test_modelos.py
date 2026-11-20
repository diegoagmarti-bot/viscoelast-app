"""Conferência das soluções analíticas com resultados conhecidos da literatura."""
import numpy as np
import sympy as sp

import modelos as m

V = {m.E: 2000.0, m.eta: 10000.0, m.Einf: 1000.0, m.tau: 5.0, m.sig0: 10.0, m.eps0: 0.005}


def iguais(a, b):
    return sp.simplify(a - b) == 0


def test_fluencia_maxwell():
    J = m.solucao_degrau("Maxwell", "fluencia")["saida_t"] / m.sig0
    assert iguais(J, 1 / m.E + m.t / (m.E * m.tau))


def test_fluencia_kelvin_voigt():
    J = m.solucao_degrau("Kelvin-Voigt", "fluencia")["saida_t"] / m.sig0
    assert iguais(J, (1 - sp.exp(-m.t / m.tau)) / m.E)


def test_relaxacao_slp():
    G = m.solucao_degrau("Sólido Linear Padrão", "relaxacao")["saida_t"] / m.eps0
    assert iguais(G, m.Einf + m.E * sp.exp(-m.t / m.tau))


def test_limites_slp():
    sol = m.solucao_degrau("Sólido Linear Padrão", "fluencia")
    assert iguais(sol["inicial"], m.sig0 / (m.E + m.Einf))
    assert iguais(sol["final"], m.sig0 / m.Einf)


def test_modulos_dinamicos():
    wt = m.w * m.tau
    mx = m.solucao_senoidal("Maxwell")
    assert iguais(mx["E_arm"], m.E * wt**2 / (1 + wt**2))
    assert iguais(mx["E_perda"], m.E * wt / (1 + wt**2))
    kv = m.solucao_senoidal("Kelvin-Voigt")
    assert iguais(kv["E_arm"], m.E) and iguais(kv["E_perda"], m.E * wt)


def test_senoidal_tende_ao_regime_permanente():
    omega = 0.5
    tt = np.linspace(100, 120, 50)  # t >> tau
    for nome in m.MODELOS:
        sig, _ = m.simular(nome, "senoidal", V, tt, omega=omega)
        sol = m.solucao_senoidal(nome)
        Ea = float(sol["E_arm"].subs(V).subs(m.w, omega))
        Ep = float(sol["E_perda"].subs(V).subs(m.w, omega))
        regime = V[m.eps0] * (Ea * np.sin(omega * tt) + Ep * np.cos(omega * tt))
        assert np.allclose(sig, regime, rtol=1e-6, atol=1e-9)


def test_boltzmann_reproduz_fluencia_e_recuperacao():
    tt = np.linspace(0, 30, 1200)
    t1 = 15.0
    for nome in m.MODELOS:
        _, ref = m.simular(nome, "recuperacao", V, tt, t_rem=t1)
        hist = ((0.0, t1, t1, 30.0), (10.0, 10.0, 0.0, 0.0))
        _, eps = m.simular(nome, "hist_tensao", V, tt, historico=hist)
        assert np.allclose(eps, ref, rtol=1e-9, atol=1e-12)


def test_rampa_de_deformacao_maxwell():
    # rampa de taxa r: sigma = eta * r * (1 - exp(-t/tau)) durante a rampa
    tt = np.linspace(0, 10, 500)
    r = 0.001
    hist = ((0.0, 10.0), (0.0, 10.0 * r))
    sig, eps = m.simular("Maxwell", "hist_deformacao", V, tt, historico=hist)
    assert np.allclose(sig, V[m.eta] * r * (1 - np.exp(-tt / V[m.tau])))
    assert np.allclose(eps, r * tt)


def test_rampa_de_deformacao_kelvin_voigt_inclui_amortecedor():
    tt = np.linspace(0.01, 9.99, 500)  # durante a rampa; em t = 10 s o amortecedor para
    r = 0.001
    sig, _ = m.simular("Kelvin-Voigt", "hist_deformacao", V, tt, historico=((0.0, 10.0), (0.0, 10.0 * r)))
    assert np.allclose(sig, V[m.E] * r * tt + V[m.eta] * r)


def test_trechos():
    degraus, rampas = m.trechos(((0.0, 5.0, 15.0, 15.0, 30.0), (0.0, 10.0, 10.0, 0.0, 0.0)))
    assert degraus == [(15.0, -10.0)]
    assert rampas == [(0.0, 5.0, 2.0)]


def test_energia_conserva_trabalho_externo():
    # trabalho externo = energia armazenada nas molas + dissipada no amortecedor
    tt = np.linspace(0, 30, 3000)
    hist = ((0.0, 10.0, 20.0, 30.0), (0.0, 10.0, 0.0, 0.0))
    for nome in m.MODELOS:
        sig, eps = m.simular(nome, "hist_tensao", V, tt, historico=hist)
        total = sum(m.energias(m.componentes(nome, sig, eps, V)).values())
        trabalho = np.concatenate([[0.0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * np.diff(eps))])
        assert np.allclose(total, trabalho, atol=1e-6 * np.max(np.abs(trabalho)))
