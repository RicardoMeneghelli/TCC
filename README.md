# TCC – Detecção de requisições HTTP maliciosas com Isolation Forest

Experimentos do artigo "Detecção de requisições HTTP maliciosas com Isolation Forest:
uma avaliação no conjunto CSIC 2010" (UNESC).

## Conteúdo

- `experimento_csic.py` – extração de 14 atributos por requisição e avaliação do
  Isolation Forest em três cenários (A: ataques em 5% do tráfego; B: treino só com
  tráfego normal; C: conjunto completo). Gera `resultados_csic.json`.
- `experimento_comparacao.py` – compara Isolation Forest, Local Outlier Factor,
  One-Class SVM e Random Forest (supervisionado). Gera `resultados_comparacao.json`
  e `roc_curvas.png`.
- `resultados_csic.json`, `resultados_comparacao.json`, `roc_curvas.png` – saídas
  usadas no artigo.

## Como reproduzir

1. Instale as dependências: `pip install -r requirements.txt`
2. Baixe o conjunto de dados CSIC 2010 (HTTP dataset CSIC 2010, do CSIC, Espanha) e
   salve o arquivo CSV como `csic_database.csv` nesta pasta. O arquivo **não** está
   neste repositório; consulte a licença e as condições de uso na fonte original.
3. Execute:
   ```
   python experimento_csic.py
   python experimento_comparacao.py
   ```

Os experimentos usam sementes fixas. Os resultados do artigo foram obtidos com
Python 3.12, scikit-learn 1.8.0 e pandas 3.0.2, em um único núcleo, sem GPU.
Versões diferentes das bibliotecas podem alterar levemente os números.

## Observação metodológica

No CSIC 2010, cada requisição GET com parâmetros tem um POST equivalente. A divisão
entre treino e teste é feita por grupos (mesmo caminho e mesma carga útil) para
evitar vazamento de informação entre as duas partes.
