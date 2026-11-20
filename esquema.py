"""Desenho animado do modelo reológico (molas e amortecedores) com plotly."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COR_MOLA = "#1F4E79"
COR_AMORT = "#9C3D2E"
COR_ESTRUTURA = "#5F6B7A"
L0, A = 1.3, 0.7  # comprimento de referência e amplitude visual de cada elemento


def mola(x0, x1, y, n=8, h=0.13):
    lead = 0.12
    xs = np.linspace(x0 + lead, x1 - lead, 2 * n + 1)
    ys = y + h * np.array([0] + [(-1) ** k for k in range(2 * n - 1)] + [0])
    return [x0, *xs, x1], [y, *ys, y]


def amortecedor(x0, x1, y, h=0.15):
    c0, c1 = x0 + 0.1, x0 + 0.6  # cilindro de comprimento fixo
    curso = np.clip((x1 - x0 - (L0 - A)) / (2 * A), 0, 1)
    xp = c0 + 0.05 + 0.4 * curso  # pistão percorre o cilindro conforme o alongamento
    xs = [x0, c0, None, c1, c0, c0, c1, None, xp, xp, None, xp, x1]
    ys = [y, y, None, y + h, y + h, y - h, y - h, None, y + 0.75 * h, y - 0.75 * h, None, y, y]
    return xs, ys


def _comprimento(e, escala):
    return L0 + A * e / escala


def geometria(nome, partes):
    """Lista de (x, y, cor, rótulo, x_rótulo, y_rótulo) e posição da placa móvel."""
    elems = []
    if nome == "Maxwell":
        x1 = partes[0]
        x2 = x1 + partes[1]
        elems.append((*mola(0, x1, 0), COR_MOLA, "E", x1 / 2, 0.32))
        elems.append((*amortecedor(x1, x2, 0), COR_AMORT, "η", (x1 + x2) / 2, 0.32))
        return elems, x2, (-0.25, 0.25)
    if nome == "Kelvin-Voigt":
        x1 = partes[0]
        elems.append((*mola(0, x1, 0.35), COR_MOLA, "E", x1 / 2, 0.65))
        elems.append((*amortecedor(0, x1, -0.35), COR_AMORT, "η", x1 / 2, -0.7))
        return elems, x1, (-0.55, 0.55)
    x1 = partes[1]
    x2 = x1 + partes[2]
    elems.append((*mola(0, x2, 0.35), COR_MOLA, "E∞", x2 / 2, 0.65))
    elems.append((*mola(0, x1, -0.35), COR_MOLA, "E", x1 / 2, -0.05))
    elems.append((*amortecedor(x1, x2, -0.35), COR_AMORT, "η", (x1 + x2) / 2, -0.05))
    return elems, x2, (-0.55, 0.55)


def tracos_esquema(nome, partes, forca):
    elems, xp, (ya, yb) = geometria(nome, partes)
    tracos = []
    for xs, ys, cor, rot, xr, yr in elems:
        tracos.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cor, width=2.5),
                                 hoverinfo="skip"))
        tracos.append(go.Scatter(x=[xr], y=[yr], mode="text", text=[rot],
                                 textfont=dict(size=15, color=cor), hoverinfo="skip"))
    tracos.append(go.Scatter(x=[xp, xp], y=[ya, yb], mode="lines",
                             line=dict(color=COR_ESTRUTURA, width=6), hoverinfo="skip"))
    # seta da tensão aplicada (ponta sempre no segundo ponto)
    ponta = xp + 0.15 + 0.6 * abs(forca)
    if forca >= 0:
        xs, simbolo = [xp + 0.15, ponta], "triangle-right"
    else:
        xs, simbolo = [ponta, xp + 0.15], "triangle-left"
    tracos.append(go.Scatter(x=xs, y=[0, 0], mode="lines+markers",
                             line=dict(color="#2B2B2B", width=2),
                             marker=dict(symbol=simbolo, size=[0, 12 if abs(forca) > 0.02 else 0], color="#2B2B2B"),
                             hoverinfo="skip"))
    return tracos


def figura_animada(nome, tt, partes_hist, forca_hist, resposta, rotulo_resp, n_quadros=80):
    """Figura com o esquema animado (acima) e a curva de resposta com marcador (abaixo).

    partes_hist: lista de vetores de deformação de cada elemento, na ordem da geometria.
    forca_hist: tensão aplicada normalizada em [-1, 1].
    """
    # cada elemento em sua própria escala, para que deformações pequenas também apareçam
    maximo = max(np.max(np.abs(p)) for p in partes_hist) or 1.0
    escalas = [max(np.max(np.abs(p)), 1e-6 * maximo) for p in partes_hist]
    idx = np.linspace(0, len(tt) - 1, n_quadros).astype(int)

    def partes_em(i):
        return [_comprimento(p[i], e) for p, e in zip(partes_hist, escalas)]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.42, 0.58], vertical_spacing=0.1)
    # parede fixa
    fig.add_trace(go.Scatter(x=[0, 0], y=[-0.75, 0.75], mode="lines",
                             line=dict(color=COR_ESTRUTURA, width=6), hoverinfo="skip"), 1, 1)
    hachura_x, hachura_y = [], []
    for y in np.linspace(-0.7, 0.7, 8):
        hachura_x += [0, -0.12, None]
        hachura_y += [y, y - 0.12, None]
    fig.add_trace(go.Scatter(x=hachura_x, y=hachura_y, mode="lines",
                             line=dict(color=COR_ESTRUTURA, width=1), hoverinfo="skip"), 1, 1)

    iniciais = tracos_esquema(nome, partes_em(idx[0]), forca_hist[idx[0]])
    primeiro_movel = len(fig.data)
    for tr in iniciais:
        fig.add_trace(tr, 1, 1)
    fig.add_trace(go.Scatter(x=tt, y=resposta, mode="lines", line=dict(color=COR_MOLA, width=2.5),
                             name=rotulo_resp, hovertemplate="t = %{x:.3g} s<br>%{y:.4g}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=[tt[idx[0]]], y=[resposta[idx[0]]], mode="markers",
                             marker=dict(size=11, color=COR_AMORT), hoverinfo="skip"), 2, 1)
    moveis = list(range(primeiro_movel, primeiro_movel + len(iniciais))) + [len(fig.data) - 1]

    quadros = []
    for k, i in enumerate(idx):
        dados = tracos_esquema(nome, partes_em(i), forca_hist[i])
        dados.append(go.Scatter(x=[tt[i]], y=[resposta[i]]))
        quadros.append(go.Frame(data=dados, traces=moveis, name=str(k)))
    fig.frames = quadros

    x_max = 2 * (L0 + A) + 1.0 if nome != "Kelvin-Voigt" else L0 + A + 1.0
    fig.update_xaxes(visible=False, range=[-0.3, x_max], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.95, 0.95], row=1, col=1)
    fig.update_xaxes(title_text="Tempo, t (s)", showgrid=True, gridcolor="#E6E9EE", row=2, col=1)
    fig.update_yaxes(title_text=rotulo_resp, showgrid=True, gridcolor="#E6E9EE", row=2, col=1)

    opcoes = dict(frame=dict(duration=60, redraw=False), transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        height=620, showlegend=False, template="plotly_white",
        margin=dict(l=60, r=20, t=20, b=40),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.08, xanchor="left", yanchor="top",
            pad=dict(r=10, t=10), showactive=False,
            buttons=[dict(label="Reproduzir", method="animate", args=[None, {**opcoes, "fromcurrent": True}]),
                     dict(label="Pausar", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])],
        )],
        sliders=[dict(
            active=0, x=0.2, y=-0.06, len=0.8, currentvalue=dict(prefix="t = ", suffix=" s"),
            pad=dict(t=10), steps=[dict(method="animate", label=f"{tt[i]:.3g}",
                                        args=[[str(k)], opcoes]) for k, i in enumerate(idx)],
        )],
    )
    return fig
