# Viscoelasticidade Linear Unidimensional

Aplicativo web (Streamlit) para resolver as equações constitutivas diferenciais e integrais da
viscoelasticidade linear 1D.

## Funcionalidades

- Modelos reológicos: **Maxwell**, **Kelvin-Voigt** e **Sólido Linear Padrão** (Zener).
- Históricos de carregamento: fluência (tensão constante), relaxação (deformação constante),
  fluência e recuperação, histórico de tensão personalizado (tabela de pontos, resolvido pela
  integral hereditária de Boltzmann) e carga senoidal.
- Comparação dos três modelos sob o mesmo carregamento.
- Animação do modelo mecânico: molas e amortecedores se deformam ao longo do tempo,
  sincronizados com a curva de resposta.
- Deformação e tensão em cada componente (mola e amortecedor).
- Solução analítica passo a passo com `sympy` (transformada de Laplace, frações parciais,
  inversa, funções de fluência e relaxação, verificação pelos limites, forma integral) antes da
  substituição numérica. Na carga senoidal, a solução completa inclui o transitório.
- Análise dinâmica (DMTA): módulo de armazenamento E′, módulo de perda E″, tan δ, laço de
  histerese, energia dissipada por ciclo e varredura em frequência.
- Exportação dos resultados em CSV.

## Estrutura

| Arquivo | Conteúdo |
| --- | --- |
| `app.py` | Interface Streamlit |
| `modelos.py` | Modelos, soluções simbólicas (sympy) e numéricas |
| `esquema.py` | Desenho animado de molas e amortecedores (plotly) |
| `tests/` | Testes das soluções analíticas (pytest) |
| `requirements.txt` | Dependências para o Streamlit Community Cloud |

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Testes

```bash
pip install pytest
pytest
```

## Convenções

- Unidades: tensão e módulos em MPa, viscosidade em MPa·s, tempo em s. A deformação é exibida
  em % nos gráficos e em mm/mm na solução analítica.
- Os três modelos usam a mesma mola E e o mesmo amortecedor η, com τ = η/E.
- Forma operacional das equações: `P(D) σ = Q(D) ε`, com `D = d/dt`.
- Sólido Linear Padrão na forma de Maxwell: mola E∞ em paralelo com um braço de Maxwell (E, η).
