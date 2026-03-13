# Implementação LeNet-5 com PyTorch

Implementação da arquitetura clássica **LeNet-5** proposta por Yann LeCun et al. (1998) no artigo [*Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), treinada no dataset MNIST de dígitos manuscritos.

---

## Arquitetura

| Camada | Tipo | Filtros / Unidades | Kernel | Ativação |
|--------|------|--------------------|--------|----------|
| C1 | Conv2d | 6 | 5×5 | ReLU |
| S2 | MaxPool2d | — | 2×2 | — |
| C3 | Conv2d | 16 | 5×5 | ReLU |
| S4 | MaxPool2d | — | 2×2 | — |
| C5 | Conv2d | 120 | 5×5 | ReLU |
| F6 | Linear | 84 | — | ReLU |
| Saída | Linear | 10 | — | — |

> **Observação:** `padding=2` é aplicado na camada C1 para compatibilidade com imagens 28×28 do MNIST, equivalente à entrada original de 32×32.

---

## Estrutura do Projeto

```
lenet/
├── data/                    # Dataset MNIST (baixado automaticamente)
├── weights/
│   └── lenet5_mnist.pth     # Pesos salvos do modelo
├── lenet5.ipynb             # Notebook principal
└── README.md
```

---

## Conteúdo do Notebook

O notebook está organizado nas seguintes seções:

1. **Visualização dos Filtros Iniciais** — Plota os 6 filtros da `conv1` com pesos aleatórios antes do treinamento
2. **Carregamento e Visualização do MNIST** — Baixa o dataset e exibe imagens de exemplo
3. **Visualização dos Feature Maps** — Mostra as ativações da `conv1` em uma imagem de exemplo
4. **Treinamento da Rede** — Loop de treinamento completo com CrossEntropyLoss e otimizador Adam
5. **Avaliação do Modelo** — Calcula a acurácia no conjunto de teste e coleta amostras classificadas incorretamente
6. **Visualização dos Erros** — Plota imagens onde o modelo errou a previsão
7. **Salvamento e Carregamento do Modelo** — Salva o `state_dict` em disco e demonstra como recarregá-lo

---

## Requisitos

- Python 3.11+
- PyTorch
- torchvision
- matplotlib

Instale todas as dependências:

```bash
python -m pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cpu
```

> Para suporte a GPU, substitua `cpu` pela sua versão do CUDA (ex: `cu121`).

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/lenet.git
cd lenet

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Instale as dependências
python -m pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cpu

# 4. Abra o notebook
jupyter notebook lenet5.ipynb
```

---

## Treinamento

O modelo é treinado no conjunto de treino do MNIST (60.000 imagens) com os seguintes hiperparâmetros:

| Hiperparâmetro | Valor |
|----------------|-------|
| Otimizador | Adam |
| Taxa de Aprendizado | 0.001 |
| Tamanho do Batch | 64 |
| Épocas | 5 |
| Função de Perda | CrossEntropyLoss |

---

## Resultados

Após 5 épocas de treinamento em CPU, o modelo atinge aproximadamente **98% de acurácia** no conjunto de teste do MNIST (10.000 imagens).

---

## Referência

> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
> *Gradient-based learning applied to document recognition.*
> Proceedings of the IEEE, 86(11):2278–2324, November 1998.
