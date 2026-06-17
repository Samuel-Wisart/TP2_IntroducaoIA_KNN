# TP2 - Aprendizado de Máquina

Projeto em Python para implementar e comparar os algoritmos `kNN` e `k-Means` usando a base `NBA Rookie Stats`.

## Estrutura

- `src/`: código principal do projeto.
- `scripts/`: executáveis para rodar os experimentos.
- `notebooks/`: análise visual e geração de gráficos para o relatório.
- `results/`: saídas geradas pelos experimentos.
- `reports/`: espaço para o relatório final e figuras.

## Dependências

```bash
pip install -r requirements.txt
```

## Como executar

Rodar todos os experimentos:

```bash
python scripts/run_all_experiments.py
```

Rodar apenas kNN:

```bash
python scripts/run_knn_experiments.py
```

Rodar apenas k-Means:

```bash
python scripts/run_kmeans_experiments.py
```

Abrir o notebook de análise para gerar gráficos:

```bash
jupyter notebook notebooks/analise_resultados.ipynb
```

## Observações

- O projeto usa padronização dos atributos antes de calcular distâncias.
- Os resultados são salvos em `results/`.
- O notebook foi criado para apoiar a análise e a escrita do relatório.
