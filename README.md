# Frac Pump Failure Forecasting

## PT-BR

### Visão rápida
Este projeto mostra como estruturar um pipeline de **previsão de falha para bombas de fraturamento** usando sinais de telemetria operacional. O objetivo não é apenas detectar quando a bomba já está ruim, mas estimar o risco de falha na próxima janela operacional para permitir planejamento de intervenção, troca de componente e priorização de ativos críticos.

### Problema de negócio
Em operações de completions e pressure pumping, falhas de bomba têm impacto direto em:
- continuidade da etapa;
- custo de manutenção não planejada;
- uso da frota reserva;
- produtividade do spread;
- risco operacional em campo.

Por isso, um modelo útil nesse cenário precisa responder:
- qual bomba está mais próxima de falhar;
- quais sinais explicam essa deterioração;
- quais unidades devem ser priorizadas antes da próxima janela crítica.

### Base pública de referência
O framing técnico do projeto usa como referência o **AI4I 2020 Predictive Maintenance Dataset**, da UCI, porque ele é uma base pública conhecida para manutenção preditiva industrial. Para manter o projeto reprodutível e alinhado ao domínio de frac pumps, o runtime usa uma amostra local `AI4I-style` adaptada para telemetria de bombas de fraturamento.

Referência oficial:
- [UCI - AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### O que o projeto faz
1. Gera uma base local de telemetria por bomba.
2. Treina um classificador para prever `failure_next_window`.
3. Pontua a última janela observada de cada bomba.
4. Converte a probabilidade em `forecast_band` e `health_score`.
5. Exporta um resumo da frota pronto para decisão operacional.

### Estrutura do projeto
- `main.py`: entry point local.
- `src/sample_data.py`: gera a base sintética inspirada em manutenção industrial.
- `src/modeling.py`: treina o modelo e exporta o ranking final de risco.
- `tests/test_project.py`: valida o contrato mínimo do pipeline.
- `data/raw/public_dataset_reference.json`: referência da base pública.
- `data/processed/frac_pump_scored_windows.csv`: holdout pontuado.
- `data/processed/pump_failure_forecast_summary.csv`: snapshot final por bomba.
- `data/processed/frac_pump_failure_forecast_report.json`: relatório consolidado.

### Variáveis do dataset
- `pump_id`: identificador da bomba.
- `window_number`: posição temporal da janela observada.
- `discharge_pressure`: pressão de descarga.
- `suction_pressure`: pressão de sucção.
- `fluid_rate`: taxa de bombeio.
- `vibration`: vibração mecânica.
- `power_end_temperature`: temperatura do power end.
- `lube_pressure`: pressão de lubrificação.
- `stroke_rate`: frequência operacional da bomba.
- `valve_noise_index`: proxy de ruído e desgaste de válvula.
- `failure_next_window`: alvo supervisionado de previsão.

### Modelagem
O pipeline usa:
- imputação para variáveis numéricas e categóricas;
- `OneHotEncoder` para `pump_id`;
- `RandomForestClassifier` para aprender relações não lineares entre degradação, carga e risco futuro.

Essa modelagem é adequada para o MVP porque:
- lida bem com sinais heterogêneos;
- capta interações entre pressão, vibração, temperatura e desgaste;
- gera bom baseline sem tuning pesado;
- é fácil de explicar em entrevista e discussão arquitetural.

### Saídas operacionais
O projeto gera duas visões:

**1. Scored windows**
- previsão sobre o holdout;
- útil para validar métricas offline.

**2. Forecast summary**
- última janela por bomba;
- `predicted_probability`;
- `forecast_band` em `stable`, `elevated` ou `critical`;
- `health_score` em escala de `0` a `100`.

Essa segunda visão é a mais próxima do uso real em uma control tower de manutenção.

### Resultados atuais
- `dataset_source = frac_pump_failure_sample_ai4i_style`
- `row_count = 635`
- `pump_count = 8`
- `positive_rate = 0.2331`
- `roc_auc = 0.9171`
- `average_precision = 0.8725`
- `f1 = 0.7792`
- `critical_pumps = 6`

### Como executar
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### Do básico ao técnico
No nível mais básico, este projeto é um classificador supervisionado de risco de falha.

No nível intermediário, ele é um pipeline de **forecasting operacional de falha** com saída por bomba.

No nível avançado, ele permite discutir:
- predição de falha em janela futura;
- batch scoring versus atualização near real-time;
- governança de sinais industriais;
- monitoramento de drift;
- escalabilidade por spread, poço ou região.

### Batch vs stream
- `batch`:
  - recomputar features históricas;
  - retreinar modelo;
  - recalcular ranking completo da frota;
  - consolidar relatórios de turno ou campanha.

- `stream`:
  - atualizar risco quando chega nova telemetria;
  - disparar alerta para bomba crítica;
  - alimentar painel operacional em baixa latência.

Trade-off:
- batch é mais simples e reprodutível;
- stream entrega menor latência, mas aumenta a complexidade de operação, observabilidade e controle de eventos.

### Governança e monitoramento
Em produção, este tipo de pipeline pede:
- validação de ranges físicos dos sensores;
- controle de missing e atraso de leitura;
- lineage entre telemetria, features e score final;
- monitoramento de drift e degradação de performance;
- rastreabilidade do modelo que originou o forecast.

### Limitações
- o runtime usa uma amostra local inspirada em datasets públicos de manutenção industrial;
- a validação é offline;
- o rótulo de falha futura é um proxy supervisionado de risco, não uma ordem de manutenção real.

## EN

### Quick overview
This project structures a **frac pump failure forecasting** workflow using operational telemetry. The goal is to estimate the probability of failure in the next operating window so maintenance teams can prioritize critical pumps before downtime happens.

### Public dataset framing
The project is technically framed around the **AI4I 2020 Predictive Maintenance Dataset** from UCI. Runtime execution uses a compact local `AI4I-style` telemetry sample adapted to a frac pump setting for deterministic execution.

### What the project does
1. Generates a local pump telemetry sample.
2. Trains a classifier for `failure_next_window`.
3. Scores the latest observed window of each pump.
4. Converts probabilities into `forecast_band` and `health_score`.
5. Exports a fleet summary for operational prioritization.

### Current results
- `dataset_source = frac_pump_failure_sample_ai4i_style`
- `row_count = 635`
- `pump_count = 8`
- `positive_rate = 0.2331`
- `roc_auc = 0.9171`
- `average_precision = 0.8725`
- `f1 = 0.7792`
- `critical_pumps = 6`

### Run locally
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### Advanced discussion points
This repository is also useful to discuss:
- future-window failure forecasting;
- batch versus near-real-time scoring;
- industrial telemetry governance;
- drift monitoring;
- scaling predictive maintenance across spreads and regions.
