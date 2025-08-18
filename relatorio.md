[relatorio_tecnico_previsao_de_churn.md](https://github.com/user-attachments/files/21845474/relatorio_tecnico_previsao_de_churn.md)
# Relatório Técnico — Previsão de Churn

## 1) Objetivo

Desenvolver, comparar e selecionar modelos de Machine Learning para prever **churn** (classe 1), priorizando **recall da classe 1** como métrica de seleção, de modo a reduzir a perda de clientes não detectados.

---

## 2) Dados, Pré‑processamento e Seleção de Features

**Alvo:** `Churn` (0 = permanece, 1 = churn).

**Split:** treino/teste estratificado.

**Balanceamento:** aplicado **apenas no treino** (ex.: SMOTE), evitando vazamento para o teste.

**Normalização:** aplicada para modelos baseados em distância (KNN).

**Critério de seleção:** **Permutation Importance** com *scoring* = `recall` para medir a contribuição de cada variável na detecção de churn (não informa direção do efeito, apenas relevância para a métrica).

**Features removidas por baixa relevância (solicitado):**

- `internet.OnlineBackup_Yes`
- `internet.DeviceProtection_Yes`
- `account.PaymentMethod_Credit card (automatic)`

> Observação: valores de importância **negativos** indicam que a feature pode introduzir ruído para o *recall* do modelo e são boas candidatas a exclusão. Recomenda-se reavaliar com `permutation_importance` e/ou SHAP antes de consolidar o *feature set* final.

**Principais drivers (maior contribuição para *****recall***** no teu KNN):**

- `customer.tenure`
- `customer.Partner_Yes`
- `internet.OnlineSecurity_Yes`
- `account.Contract_Two year`
- `account.Contract_One year`

---

## 3) Modelos Avaliados e Métricas

Foram avaliados três modelos. Abaixo, as métricas reportadas no conjunto de **teste**:

| Modelo                             | Accuracy | Precision (classe 1) | Recall (classe 1) | F1 (classe 1) |
| ---------------------------------- | -------- | -------------------- | ----------------- | ------------- |
| **Modelo 1 - arvore**              | 0.73     | 0.50                 | 0.56              | 0.53          |
| **Modelo 2 — Regressão Logística** | 0.71     | 0.47                 | **0.69**          | 0.56          |
| **Modelo 3 - logistic**            | **0.76** | 0.54                 | 0.66              | **0.59**      |

**Critério de escolha para churn:** maximizar **recall da classe 1**. Assim, o **Modelo 2 (Regressão Logística)** é o mais indicado, pois identifica a maior fração de clientes que efetivamente irão evadir (recall = 0.69), ainda que com leve perda de precisão/acurácia global.

---

## 4) Interpretação dos Fatores de Evasão

> A *Permutation Importance* quantifica **o quanto cada variável contribui para o *****recall*** do modelo; **não** determina se o efeito é positivo ou negativo. Para direção do efeito, usar **PDP/ICE** ou **SHAP**.

**Insights práticos (a validar com análises direcionais):**

- ``** (tempo de cliente):** clientes com menor tempo típico tendem a apresentar maior risco. Estratégias de onboarding e *early engagement* são críticas.
- **Tipo de contrato (**``**/**``**):** maior duração de contrato costuma estar associada a menor churn em comparação a mensalidade (month‑to‑month). Incentivos de fidelização ajudam.
- ``**:** adesão a serviços de segurança e valor agregado correlaciona com retenção superior; *bundles* e *upsell* podem reduzir churn.
- ``**:** perfis com rede de suporte (ex.: parceiro na conta) tendem a ser mais estáveis; ofertas *family/duo* podem reforçar permanência.

---

## 5) Recomendações de Retenção

**Ações táticas imediatas (focadas em alto risco):**

1. **Onboarding intensivo para novos clientes (baixa **``**)**: sequência de mensagens de valor, check‑ins proativos nos 30/60/90 dias, *quick wins* e benefícios iniciais.
2. **Incentivos para migração de contrato**: descontos escalonados para upgrade de planos mensais → anuais/bianuais; incluir *free trials* de serviços de segurança.
3. **Pacotes de valor agregado**: promover **OnlineSecurity** e serviços de suporte como *bundle* para perfis com maior propensão ao churn.
4. **Campanhas direcionadas por segmentos**: usar o modelo para gerar *scores* semanais; priorizar o top‑X% de risco para ofertas de retenção (ex.: crédito, dados extra, melhoria de velocidade).
5. **Monitoramento e *****A/B testing***: medir *uplift* por campanha de retenção vs. grupo controle para provar ROI.

**Aprimoramentos estruturais (médio prazo):**

- **Ajuste de limiar de decisão** para calibrar a troca *recall × precisão* conforme orçamento de retenção.
- **Custo‑sensibilidade** nas perdas (ponderar *false negative* > *false positive* em treinamento/thresholding).
- **Revisitar features com importância negativa** e removê‑las se persistirem como ruído.
- **Explicabilidade** com **SHAP** para confirmar direção dos efeitos e comunicar ao negócio.

---

## 6) Conclusão

- **Modelo recomendado:** **Regressão Logística** (maior *recall* da classe 1).
- **Drivers operacionais:** `customer.tenure`, tipo de contrato, adesão a `OnlineSecurity`, condição `Partner_Yes`.
- **Plano de retenção:** foco em clientes novos, *bundles* de segurança/valor, migração para contratos mais longos, e campanhas dirigidas por *score* de risco.

---

## 7) Anexo — Código de Referência (remoção de features + pipeline)

> Ajuste os nomes das colunas e da variável‑alvo conforme o teu *notebook*. Este exemplo mantém o balanceamento apenas no treino e otimiza o K do KNN visando **recall**.

```python
# === Setup ===
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# === Dados ===
# df: DataFrame com features já numéricas/one-hot e alvo 'Churn'
X = df.drop('Churn', axis=1)
y = df['Churn']

# === Remover features pouco relevantes (solicitado) ===
cols_to_drop = [
    'internet.OnlineBackup_Yes',
    'internet.DeviceProtection_Yes',
    'account.PaymentMethod_Credit card (automatic)'
]
X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Balancear apenas treino ===
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# === KNN: otimizar K para maximizar recall ===
recall_scorer = make_scorer(recall_score)
knn_pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {"knn__n_neighbors": [3,5,7,9,11], "knn__weights": ["uniform","distance"]}
knn_cv = GridSearchCV(knn_pipe, param_grid, scoring=recall_scorer, cv=5, n_jobs=-1)
knn_cv.fit(X_train_bal, y_train_bal)
knn_best = knn_cv.best_estimator_

# Avaliação KNN
print("KNN — melhor config:", knn_cv.best_params_)
y_pred_knn = knn_best.predict(X_test)
print("\nKNN — Teste:\n", classification_report(y_test, y_pred_knn, digits=3))
print(confusion_matrix(y_test, y_pred_knn))

# === Regressão Logística (baseline recomendado) ===
logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
logreg.fit(X_train_bal, y_train_bal)
y_pred_lr = logreg.predict(X_test)
print("\nRegressão Logística — Teste:\n", classification_report(y_test, y_pred_lr, digits=3))
print(confusion_matrix(y_test, y_pred_lr))

# === Decision Tree (comparativo) ===
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train_bal, y_train_bal)
y_pred_tree = clf_tree.predict(X_test)
print("\nDecision Tree — Teste:\n", classification_report(y_test, y_pred_tree, digits=3))
print(confusion_matrix(y_test, y_pred_tree))

# === Sumário prático ===
# Monte um DataFrame com métricas por modelo (recall/precision/F1 da classe 1, accuracy)
from sklearn.metrics import accuracy_score, precision_score, f1_score

def metrics_row(y_true, y_pred, name):
    return {
        'Modelo': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision_1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1_1': f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    }

summary = pd.DataFrame([
    metrics_row(y_test, y_pred_lr, 'LogisticRegression'),
    metrics_row(y_test, y_pred_knn, 'KNN(best)'),
    metrics_row(y_test, y_pred_tree, 'DecisionTree')
]).sort_values('Recall_1', ascending=False)
print("\nResumo de métricas (ordenado por Recall_1):\n", summary)
```

---

## 8) Próximos Passos Técnicos

- **Calibração de probabilidade** e ajuste de limiar para alinhar com orçamento de retenção.
- **Cost‑sensitive learning** ou *class weights* para reforçar penalização de falsos negativos.
- **Explainability** com SHAP (direção e magnitude do efeito por feature e por indivíduo).
- **Monitoramento em produção** (drift, *retraining window*, *champion/challenger*).

