import numpy as np
import matplotlib.pyplot as plt

class Perceptron:     
    def __init__(self,entradas,esperados):
        self.entradas = entradas                                            #lista com as entradas de treino, onde cada entrada tbm é uma lista com valores de cada dentrito
        self.pesos = np.random.rand(len(self.entradas[0]))                  #cria pesos inicialmente aleatorios. Assume-se q todas as entradas tem o mesmo numero de dentritos
        self.bias = np.random.rand()                                        #inicializa o vies com valor aleatorio tbm
        self.esperado = esperados                                           #lista de saidas esperadas. Notar q isso é so pra aprendizado supervisionado
        self.taxa_aprendizado = 0.1
        self.funcao = lambda s: 1 if s>=0 else 0                            #funcao degrau, para introduzir nao continuidade
        self.saida = None

    def treinar(self): 
    
        while True:                                                         #esse loop é pro neuronio ficar testando seus parametros e os recalculando, eternamente, ate acertar tudo
            erro_pesos = np.zeros(len(self.pesos))                          #aqui se inicializa os erros acumulados. O metodo sera batch 
            erro_vies = 0
            erro_absoluto = 0

            for i in range(len(self.entradas)):                             #primeiro varre todas as entradas, calculando o erro total acumulado nelas                                                                 
                soma = np.dot(self.entradas[i],self.pesos) + self.bias      
                self.saida = self.funcao(soma)                              #resultado da saida do neuronio
                    
                erro = self.esperado[i] - self.saida  
                erro_pesos += erro * np.array(self.entradas[i])  
                erro_vies += erro
                erro_absoluto += abs(erro)

            self.pesos += self.taxa_aprendizado * erro_pesos                #corrige os pesos e o bias com base no total de erro q acumulou, e depois tenta de novo, ate acertar tudo
            self.bias += self.taxa_aprendizado * erro_vies

            if erro_absoluto == 0:                                          #loop so encerra quando acertar tudo
                return self.pesos,self.bias                                 #retorna os parametros do padrao descoberto

    def prever(self,entrada):                                               #dps q o modelo ja foi construido, pode-se testa-lo com essa funcao para novas entradas
        soma = np.dot(self.pesos,entrada) + self.bias
        previsao = self.funcao(soma)
        return previsao 
    


#__________________________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
    print('---------------------------OPERADORES LÓGICOS------------------------------\n')



    #---------------------------------Operador AND----------------------------------
    print('-------------AND--------------\n')
    entradas = [[0,0],[0,1],[1,0],[1,1]]
    esperados = [0,0,0,1]
    neuronio_and = Perceptron(entradas,esperados)
    parametros = neuronio_and.treinar()

    print(f'Pesos: {parametros[0]}  /  Bias: {parametros[1]}')
    for entrada in entradas:
        previsao = neuronio_and.prever(entrada)
        print(f'{entrada} : {previsao}')

    #Gráfico 2D da reta de separaçao
    plt.figure()
    cores = ['blue' if y == 0 else 'red' for y in esperados]
    for i, entrada in enumerate(entradas):
        plt.scatter(entrada[0], entrada[1], color=cores[i], s=500)
        plt.text(entrada[0], entrada[1], str(esperados[i]), color='white', fontsize=16, ha='center', va='center', fontweight='bold')
    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = -(parametros[0][0] * x_vals + parametros[1]) / (parametros[0][1] + 1e-6)  # evita divisão por zero
    plt.plot(x_vals, y_vals, 'k--')  # reta separadora
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.title('Separação do operador AND')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    #plt.savefig('AND.png')
    plt.show()
    



    #--------------------------------Operador OR----------------------------------
    print('\n------------OR--------------\n')
    entradas = [[0,0],[0,1],[1,0],[1,1]]
    esperados = [0,1,1,1]
    neuronio_or = Perceptron(entradas,esperados)
    parametros = neuronio_or.treinar()

    print(f'Pesos: {parametros[0]}  /  Bias: {parametros[1]}')
    for entrada in entradas:
        previsao = neuronio_or.prever(entrada)
        print(f'{entrada} : {previsao}')

    #Gráfico 2D da reta de separaçao:
    plt.figure()
    cores = ['blue' if y == 0 else 'red' for y in esperados]
    for i, entrada in enumerate(entradas):
        plt.scatter(entrada[0], entrada[1], color=cores[i], s=500)
        plt.text(entrada[0], entrada[1], str(esperados[i]), color='white', fontsize=16, ha='center', va='center', fontweight='bold')
    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = -(parametros[0][0] * x_vals + parametros[1]) / (parametros[0][1] + 1e-6)  # evita divisão por zero
    plt.plot(x_vals, y_vals, 'k--')  # reta separadora
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.title('Separação do operador OR')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    #plt.savefig('OR.png')
    plt.show()

    


    #-----------------------------------Operador NOT---------------------------------------
    print('\n------------------NOT--------------\n')
    entradas = [[0],[1]]
    esperados = [1,0]
    neuronio_not = Perceptron(entradas,esperados)
    parametros = neuronio_not.treinar()

    print(f'Pesos: {parametros[0]}  /  Bias: {parametros[1]}')
    for entrada in entradas:
        previsao = neuronio_not.prever(entrada)
        print(f'{entrada} : {previsao}')

    #Gráfico 1D para NOT
    plt.figure()
    for i, entrada in enumerate(entradas):
        cor = 'blue' if esperados[i] == 0 else 'red'
        plt.scatter(entrada[0], 0, color=cor, s=500)
        plt.text(entrada[0], 0, str(esperados[i]), color='white', fontsize=16, ha='center', va='center', fontweight='bold')
    # Cálculo do ponto de decisão (limiar onde w*x + b = 0)
    x_limite = -parametros[1] / parametros[0][0]
    plt.axvline(x=x_limite, color='black', linestyle='--')
    plt.ylim(-1, 1)
    plt.xlim(-0.5, 1.5)
    plt.xlabel('Entrada')
    plt.title('Separação do operador NOT')
    plt.legend()
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    #plt.savefig('NOT.png')
    plt.show()




    #------------------------------Operador XOR---------------------------------
    print('\n-----------OPERADOR XOR--------------\n')
    entradas = [[0,0],[0,1],[1,0],[1,1]]
    esperados = [0,1,1,0]
    #mudar pra f(s) = 1 if (s>=0 and s<=1) else 0

    #Gráfico 2D: ao contrario dos outros 3, aqui o grafico sera plotado antes da reta pois os parametros da reta levam tempo infinito pra serem achados, oq faz q qlqr coisa apos eles no codigo nunca seja executado
    plt.figure()
    for i, entrada in enumerate(entradas):
        cor = 'blue' if esperados[i] == 0 else 'red'
        plt.scatter(entrada[0], entrada[1], color=cor, s=500)
        plt.text(entrada[0], entrada[1], str(esperados[i]), color='white', fontsize=16,ha='center', va='center', fontweight='bold')
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.title('Valores XOR')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    #plt.savefig('XOR.png')
    plt.show()

    neuronio_xor = Perceptron(entradas,esperados)
    parametros = neuronio_xor.treinar() 
    print(f'Pesos: {parametros[0]}  /  Bias: {parametros[1]}')
    for entrada in entradas:
        previsao = neuronio_xor.prever(entrada)
        print(f'{entrada} : {previsao}')
    
    '''
    Pode-se dar run no codigo que as primeiras 3 portas logicas rodarao normalmente, mas a ultima nunca mostrará nada na saida, pq o loop infinito do treino nunca acaba, 
    pq ele nunca atenderá ao criterio de parada q é erro_absoluto==0, pois é impossivel achar parametros q satisfaçam o XOR, ja q nao ha uma unica linha q possa separar seus pontos de (entrada,saida),
    pois repare q [0,0] da 0 e [1,1] tbm, enquanto o meio termo [0,1] ou [1,0] nao, oq o faz parecer uma parabola. Logo, é necessario mais de um neuronio para resolver
    esse problema. Em suma, XOR nao é lineramente separavel, por isso nao pode ser compreendido por apenas 1 neuronio.
    Isso sera feito em outro arquivo, e se considerará q:

    XOR(a,b) = OR(a,b) and NAND(a,b)
  
    Ve-se que xor pode ser compreendido se vc tiver camadas ocultas q antes aprendam or, nand e and. Portanto, 3 neuronios serao necessarios, sendo 2 ocultas, para
    aprender OR e NAND, e um de saida, para aprender AND. Repare q as 3 operacoes sao linearmente separaveis, é por isso que cada neuronio desses tem capacidade pra 
    compreende-los, e tbm repare q a rede n sabe oq ela vai aprender, pois os neuronios ocultos n sabem quais sao essas subfuncoes eles devem interpretar. A explicacao 
    q foi dada é apenas para se entender pq sao necessarios 3 neuronios, e n uma afirmacao de q tipos de operacao cada neuronio oculto aprendera, isso é imprevisivel
    '''