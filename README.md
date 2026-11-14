# Classificação de Textos com LLMs: Substituição do LM Head com Fine-Tuning e Transfer Learning

Projeto desenvolvido por **Thiago Alves** como entrega prática da disciplina de **Fundamentos de LLM** (Aula 9 — *Ajustando um LLM*, Prof. Paulo Lisboa de Almeida), no programa de Pós-Graduação em Inteligência Artificial Generativa da **Universidade Federal do Paraná (UFPR)**.

---

## 1. Objetivo e Motivação Arquitetural

O uso de Modelos de Linguagem (LLMs) para tarefas de classificação por meio de *prompts* apresenta duas limitações práticas: não há garantia determinística de que o modelo responderá exatamente no formato esperado, e o cálculo de probabilidades sobre todo o vocabulário (`tokenizer.vocab_size`) representa um custo computacional desnecessário para saídas categóricas.

Inspirado na abordagem *ULMFiT* (Howard & Ruder, 2018), este projeto substitui a camada final geradora de tokens (**LM Head**) por uma camada linear dedicada à tarefa de classificação. Por meio da classe `AutoModelForSequenceClassification` (Hugging Face `transformers`), o modelo pré-treinado passa a atuar como extrator de representações contextuais conectado diretamente a $N$ neurônios de saída (correspondentes às classes do dataset).

---

## 2. Dataset e Modelos (*Backbones*)

A estrutura do código foi projetada para ser agnóstica à arquitetura do modelo, permitindo comparar tanto modelos **Encoder** (atenção bidirecional, maior eficiência computacional para classificação) quanto **Decoder** (modelos autoregressivos):

* **Dataset Principal:** `ag_news` (classificação multiclasse de notícias em 4 categorias: *World*, *Sports*, *Business* e *Sci/Tech*), adaptável também para datasets binários como reviews de filmes (IMDb) e `sms_spam`.
* **Backbone Principal:** `Qwen/Qwen2.5-0.5B` (Decoder LLM leve).

---

## 3. Estrutura do Repositório

```text
├── fine_tuning_lm_head_classification.ipynb   # Notebook completo (carga, treino, avaliação e inferência)
└── README.md                                 # Documentação técnica do experimento
```

## 4. Etapas do Desenvolvimento

### Etapa 1 — Autenticação Segura no Ambiente Colab
* Uso do gerenciador nativo **Secrets** do Google Colab (`google.colab.userdata`) para carregar a chave `HF_TOKEN` com permissão de leitura (*Read*), evitando a exposição de credenciais no repositório público.

### Etapa 2 — Tokenização e Ajuste de Padding
* Configuração dinâmica de `tokenizer.pad_token` (utilizando `eos_token` como *fallback* para modelos Decoder que não possuem token de preenchimento nativo) e sincronização com `model.config.pad_token_id`.
* Truncamento controlado via `max_length` (`128` tokens para manchetes do `ag_news` e `256`–`512` tokens para textos mais longos, como críticas de filmes).

### Etapa 3 — Substituição do LM Head e Precisão Numérica (`dtype`)
* Carga do modelo via `AutoModelForSequenceClassification.from_pretrained(..., num_labels=NUM_CLASSES)`, descartando o *LM Head* original e instanciando a nova cabeça de classificação.
* Verificação automática de suporte de hardware (`torch.cuda.is_bf16_supported()`) para utilizar `torch.bfloat16` em GPUs compatíveis ou `torch.float32` na GPU T4 do Google Colab, evitando instabilidades numéricas durante o treinamento.

### Etapa 4 — Fine-Tuning vs. Transfer Learning (Congelamento de Camadas)
* Implementação da chave booleana `CONGELAR_BACKBONE`, permitindo alternar entre duas estratégias de treinamento:
  * **Fine-Tuning Completo (`CONGELAR_BACKBONE = False`):** Atualiza todos os parâmetros da rede (camadas internas do *backbone* + nova cabeça de classificação).
  * **Transfer Learning (`CONGELAR_BACKBONE = True`):** Congela os pesos do *backbone* (`requires_grad = False`) e treina exclusivamente a nova camada linear final (`classifier` / `score`), reduzindo drasticamente a quantidade de parâmetros treináveis e o tempo por época.
* Configuração do otimizador `AdamW` (`lr=2e-5`, `weight_decay=0.01`) filtrando apenas os parâmetros ativos (`requires_grad=True`):

```python
CONGELAR_BACKBONE = False  # True para treinar apenas a camada de classificação

if CONGELAR_BACKBONE:
    for name, param in model.named_parameters():
        if "classifier" not in name and "score" not in name:
            param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01
)
```

### Etapa 5 — Treinamento, Avaliação Comparativa e Inferência
* **Loop de Treinamento:** Execução ao longo de `NUM_EPOCHS = 3`, computando a perda (*loss*) e a acurácia de treino a cada época.
* **Avaliação Comparativa (Treino vs. Teste):** Uso de `model.eval()` combinado com `torch.no_grad()` para desativar camadas estocásticas (como *Dropout*) e economizar memória VRAM. A função `avaliar_modelo` calcula a perda e a acurácia nos conjuntos de treino e teste, permitindo medir a diferença de generalização (*overfitting gap*) e gerar o relatório completo por classe (`classification_report` do `scikit-learn`).

## 5. Como Executar

1. Abra o notebook `lm_head_classification.ipynb` diretamente no **Google Colab**.
2. No menu superior, acesse **Ambiente de execução $\rightarrow$ Alterar o tipo de ambiente de execução** e selecione aceleração por **GPU**.
3. No painel lateral esquerdo do Colab, abra a aba **Secrets** (ícone de chave), adicione o segredo `HF_TOKEN` com seu token *Read* do Hugging Face e habilite o acesso ao notebook.
4. Execute as células sequencialmente para realizar o download do dataset, o treinamento, a avaliação comparativa (treino vs. teste) e os testes de inferência.
