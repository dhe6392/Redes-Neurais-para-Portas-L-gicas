from sklearn.neural_network import MLPClassifier

entradas = [[0,0],[0,1],[1,0],[1,1]]
saidas = {'AND': [0,0,0,1], 'OR': [0,1,1,1], 'XOR': [0,1,1,0]}

for porta_logica,saida in saidas.items():   
    print(f'------------------{porta_logica}------------------')
    rede_neural = MLPClassifier(verbose=True,max_iter=2000,tol=0.01,learning_rate_init=0.2)
    rede_neural.fit(entradas,saida)
    previsoes = rede_neural.predict(entradas)
    print(previsoes)