import numpy as np

class Neuronio:     
    def __init__(self,entradas):
        self.entradas = np.array(entradas)                                                                          #lista com as entradas de treino, onde cada entrada tbm é uma lista com valores de cada dentrito. Converto pra array pra posteriormente realizar operacoes na funcao treinar() da classe Rede. Ate daria pra fazer com listas mesmo, mas fica lento, e como o numpy roda C por baixo dos panos, eh vantajoso usa-lo
        self.pesos = np.random.rand(len(self.entradas[0]))                                                          #cria pesos inicialmente aleatorios. Assume-se q todas os conjuntos de  entradas tem o mesmo numero de entradas
        self.bias = np.random.rand()                                                                                #inicializa o vies com valor aleatorio tbm
        self.funcao = lambda s: 1 / (1 + np.exp(-s))                                                                #funcao sigmoid. N da pra usar funcao degrau aqui pq ela nao é diferenciavel, e isso impossibilita ajustar os erros por backpropagation. So da certo pra AND, OR e NOT pq sao linearmente separaveis                          
        
    def saida(self):                                                                                                #precisa criar funcao ao inves de so criar um self.saida pq se nao ela vai ser calculada qnd chamar o perceptron e nao vai mudar depois, pois os pesos n foram atualizados. Mas dessa forma, sempre q ela for chamada ela estará atualizada com os pesos/bias novos
        return [self.funcao(np.dot(self.entradas[i],self.pesos) + self.bias) for i in range (len(self.entradas))]   #calcula as saidas iniciais pra cada entrada



class Rede:
    def __init__(self,neuronios,entradas,esperados):
        self.neuronios = neuronios
        self.esperados = esperados
        self.taxa_aprendizado = 0.2                                                                                  #taxa de aprendizado, limite de erro e limite de epocas sao a gosto, o programador q define esses valores e vai testando-os ate obter algo satisfatorio
        self.entradas = np.array(entradas)

    def treinar(self): 
                    
        erro_acumulado = 1                                                                                           #inicializando com valor aleatorio pra entrar no loop. Qualquer valor diferente de 0 serve aqui
        epoca = 0                                                                                                    #inicializando calculo do numero de etapas, para ver quantos ciclos feedforward foram necessarios para resolver o problema
        while erro_acumulado > 0.01 and epoca < 3000:                                                                #alem da condicao de erro, limita-se o numero de ciclos q a rede pode rodar, para o programa nao demorar demais. Ps: nem sempre o programa vai convergir, tem q ter um pouco de sorte. Geralmente o caso [0,0] e [0,1] (ou [1,0]) convergem rapido pra 0 e 1, enquanto [1,0] (ou [0,1]) e [1,1] convergem pro mesmo valor, q é e a media entre seus valores esperados (0.5). Por tentativa e erro, percebi q sse for pra convergir ele convergirá em menos de 3000 epocas. Se passar disso, nem com limites de 1 milhao de epocas ele resolve o problema 
            self.neuronios[0].entradas = self.entradas
            self.neuronios[1].entradas = self.entradas

            saidas_ocultas = [self.neuronios[0].saida(), self.neuronios[1].saida()]                                  #essas 3 linhas servem para pegar os resultados da camada oculta, transferir para o neuronio de saida e assim calcular a saida final da rede 
            self.neuronios[2].entradas = np.array(list(zip(*saidas_ocultas)))
            saidas_finais = self.neuronios[2].saida()   

            erro_acumulado = 0                                                                                       #erro_acumulado representa o tanto de erro que a rede comete para cada ciclo, é a soma todos os seus erros de cada entrada
            acum_pesos = [np.zeros_like(neuronio.pesos) for neuronio in self.neuronios]                              #esses dois arrays somam os fatores de correcao dos erros de cada neuronio para cada uma das entradas, e depois de rodar todas as entradas eles servirao como fator de correcao para o neuronio todo
            acum_bias = [0.0 for neuronio in self.neuronios]

            for i in range(len(saidas_finais)):                                                                      #para cada saida, esse loop compara com a saida esperadas e calcula o erro
                erro = self.esperados[i] - saidas_finais[i]      
                erro_acumulado += 0.5 * erro**2                                                                      #formula padrao do erro. Por ser quadratica, ela ja abstrai o sinal dos erros, de forma q erros opostos n se anulem na soma e a rede "pense" q acertou

                for j in range(len(self.neuronios)-1,-1,-1):                                                         #ajuste dos pesos do neuronio de saida, iniciando do neuronio 2 (camada de saida) e retrocedendo ate o 0, conforme o backpropagation
                    neuronio = self.neuronios[j]
                    soma = np.dot(neuronio.pesos,neuronio.entradas[i]) + neuronio.bias                               #calcula a soma ponderada q o neuronio faz
                    y = neuronio.funcao(soma)                                                                        #essa etapa é desnecessaria, mas calcular o y e depois a derivada ao inves de jogar essa expressao direto na derivada evita q o codigo tenha q calcular y duas vezes, ja q y aparece 2x na derivada do sigmoid. Isso otimiza um pouco o processamentp 
                    dv = y * (1-y)                                                                                   #derivada da funcao sigmoid é y*(1-y) 
                    if j == 2:                                                                                       #neuronio de saida
                        correcao_pesos_saida = self.taxa_aprendizado * erro * neuronio.entradas[i] * dv              #para cada entrada, faz a correcao para esse caso especifico
                        correcao_bias_saida = self.taxa_aprendizado * erro * dv
                        acum_pesos[j] += correcao_pesos_saida
                        acum_bias[j] += correcao_bias_saida
                    else:                                                                                            #neuronios da camada oculta                                                                                                                     
                        correcao_pesos = self.taxa_aprendizado * erro * dv * neuronio.entradas[i] * self.neuronios[2].pesos[j] 
                        correcao_bias = self.taxa_aprendizado * erro * dv * self.neuronios[2].pesos[j]
                        acum_pesos[j] += correcao_pesos
                        acum_bias[j] += correcao_bias   

            for i in range(len(self.neuronios)):                                                                     #corrige os pesos de cada neuronio
                neuronio = self.neuronios[i]
                neuronio.pesos += acum_pesos[i]                                                                 
                neuronio.bias += acum_bias[i]

            
            epoca += 1                                                                                               #aqui serve apenas para monitorar quantos ciclos de correcao a rede precisou ate aprender o padrao
        print(f'Épocas: {epoca}')

    def prever(self,entrada):                                                                                                    #essa serve para que vc pegue dados de entrada e calcule a saida final, isso depois de ja ter treinado os neuronios e calibrado os coeficientes na funcao treinar(). Em outras palavras, treinar() serve para calibrar a rede a partir de X_train e Y_train, e prever() serve para prever Y_test a partir de X_test, onde entrada é X_test e saida_final é Y_test
        saidas_ocultas = [None]*2
        saidas_ocultas[0] = self.neuronios[0].funcao(np.dot(self.neuronios[0].pesos,entrada) + self.neuronios[0].bias)           #calcula a saida do neuronio oculto 1
        saidas_ocultas[1] = self.neuronios[1].funcao(np.dot(self.neuronios[1].pesos,entrada) + self.neuronios[1].bias)           #idem para o segundo
        
        entrada_final = [saidas_ocultas[0],saidas_ocultas[1]]                                                                    #assume q a entrada do ultimo neuronio sao as saidas da camada oculta
        saida_final = self.neuronios[2].funcao(np.dot(self.neuronios[2].pesos,entrada_final) + self.neuronios[2].bias)           #resposta final produzida pela rede neural, ja com os pesos ajustados
        return saida_final

    

#__________________________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
    #------------------------Operador XOR-----------------------------
    entradas = [[0,0],[0,1],[1,0],[1,1]]                                            #poderiamos chamar de X_train tbm
    esperados = [0,1,1,0]                                                           #Y_train

    '''a logica desse algoritmo é criar uma rede com 3 neuronios, onde 2 deles aprenderao um certo padrao dos dados e o ultimo aprenderá a fazer a integracao desses. A ideia é criar uma camada oculta com esses 2 neuronios, que recebem as entradas como entrada,
    pegar a saida que eles calculam e conectar ao neuronio de saida, que pegará tais saidas como sua entrada, igual numa rede neural humana mesmo, onde vc conecta o axonio de um aos dentritos do proximo. Feito isso, o neuronio de saida calculara a saida final
    desse ciclo feedforward, comparará com a saidas esperadas, calculará o erro total desse ciclo, e com base nisso, irá corrigir os parametros dos neuronios, agora de frente pra trás, que é o backpropagation, em q o neuronio de saida corrige seus parametros

    e os da camada oculta corrigem os seus parametros com base nesses, de forma proporcional ao impacto (pesos) que eles tem no neuronio de saida, para q cada um se corrija proporcionalmente ao quanto é culpado pelo erro final.
    O algoritmo ficará nesse ciclo de feedforwards ate q o erro total de um ciclo-feedforward seja inferior ao quanto o programador julgou aceitavel
    '''

    #----------------Criando Neuronios---------------------
    neuronio_oculto1 = Neuronio(entradas)
    saida_oculta1 = neuronio_oculto1.saida()

    neuronio_oculto2 = Neuronio(entradas)
    saida_oculta2 = neuronio_oculto2.saida()

    entrada_final = list(zip(saida_oculta1,saida_oculta2))
    neuronio_saida = Neuronio(entrada_final)

    #---------------Criando a rede-------------------------
    neuronios = [neuronio_oculto1,neuronio_oculto2,neuronio_saida]
    rede = Rede(neuronios,entradas, esperados)
    rede.treinar()

    #---------------Testando ela--------------------
    for entrada in entradas:
        print(f'{entrada}: {rede.prever(entrada)}')
    
    
