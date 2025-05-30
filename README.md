# Redes Neurais para Portas Lógicas

Este projeto implementa uma rede neural simples, criada do zero em Python com NumPy, para aprender e visualizar os principais operadores lógicos (AND, OR, NOT, XOR).

## 📁 Estrutura dos Arquivos

- `Perceptron.py`: Implementa um perceptron de camada única capaz de aprender as portas **AND**, **OR** e **NOT**, que são linearmente separáveis. Também inclui uma tentativa de aplicar esse modelo ao **XOR**, que falha, pois XOR **não** é linearmente separável.
- `XOR.py`: Implementa uma rede neural com **camada oculta** e **backpropagation**, permitindo que o modelo aprenda a resolver corretamente a operação **XOR**. Também foi testado e aprovado para as outras portas.
- `AND.png`, `OR.png`, `NOT.png`: Gráficos de entrada/saída com as respectivas **retas de separação linear** aprendidas para cada porta lógica.
- `XOR.png`: Gráfico dos dados da função XOR, evidenciando a **não-linearidade** que impede sua separação por uma única reta.

## 🚀 Execução

1. **Executar o `Perceptron.py`**:
   - Treina um neurônio simples para aprender AND, OR e NOT.
   - Mostra os parâmetros aprendidos e exibe os gráficos correspondentes com a separação dos dados.

2. **Executar o `XOR.py`**:
   - Constrói uma rede com 2 neurônios na camada oculta e 1 de saída.
   - Usa a função de ativação **sigmoid** e aplica **backpropagation** para treinar a rede.
   - Resolve com sucesso a operação XOR e imprime as previsões.

## 🧠 Conceitos Envolvidos

- **Perceptron**: Algoritmo de aprendizado para dados linearmente separáveis.
- **Função degrau**: Usada em `Perceptron.py` para simular a ativação binária (0 ou 1).
- **Feedforward**: Fluxo em que cada neurônio transforma entradas em saídas, alimentando os neurônios seguintes.
- **Sigmoid + Backpropagation**: Usada em `XOR.py` para permitir ajustes contínuos dos pesos e possibilitar o aprendizado não linear.
- **Linearidade**: Apenas AND, OR e NOT são linearmente separáveis com um único neurônio.
- **Camadas ocultas**: Necessárias para resolver o XOR, pois ele exige **composição de funções não-lineares**.

## 📊 Visualizações

Cada gráfico mostra os pontos de entrada da operação lógica com suas respectivas saídas esperadas (cores e rótulos), além da reta de separação (quando possível).

Legenda:
- Azul = saída esperada 0
- Vermelho = saída esperada 1
- Linha tracejada = fronteira de decisão aprendida

## ✅ Resultados Esperados

| Operador | Acerto com 1 neurônio      | Gráfico com linha separadora? |
|----------|----------------------------|-------------------------------|
| AND      | ✅                        | ✅                            |
| OR       | ✅                        | ✅                            |
| NOT      | ✅                        | ✅                            |
| XOR      | ❌ (com 1 neurônio)       | ❌ (não-linear)               |
| XOR      | ✅ (com 3 neurônios)      | 🔁 Requer rede com backprop   |

## 📌 Observações

- O `Perceptron.py` mostra que o XOR não pode ser resolvido com um único neurônio.
- O `XOR.py` resolve o problema corretamente, simulando uma **rede neural multicamada** e mostrando como o aprendizado profundo permite capturar padrões mais complexos.
- O aprendizado em XOR.py pode não convergir dependendo da sorte na inicialização aleatória dos pesos, que pode gerar casos desfavoráveis e lentificar o treinamento.

## 📚 Requisitos

- Python 3.x
- `numpy`
- `matplotlib`

