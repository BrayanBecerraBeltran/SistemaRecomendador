import numpy as np 
import tensorflow as tf


entrada = np.array([[10,10,1,10,1],[9,9,1,9,1],[8,8,1,8,1],[7,7,1,7,1],[6,6,1,6,1],[5,5,1,5,1],[4,4,1,4,1],[3,3,1,3,1],[2,2,1,2,1],[1,1,1,1,1]], dtype=float)                          
salida = np.array([[1,0,0],[0.9, 0.05, 0.05],[0.8, 0.1, 0.1],[0.7, 0.15, 0.15],[0.6, 0.2, 0.2],[0.5, 0.25, 0.25],[0.4, 0.3, 0.3],[0.3, 0.35, 0.35],[0.2, 0.4, 0.4],[0.1, 0.45, 0.45]], dtype=float)            

capa = tf.keras.layers.Dense(units = 3, input_shape = [5], activation='sigmoid')
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), #Que tanto ajustar pesos y sesgos
    loss='mean_squared_error',
)

print("Comensando entrenamiento")
historial = modelo.fit(entrada, salida, epochs=500, verbose=False)
print("Modelo entrenado")

print("Hagamos una prediccion")
resultado = modelo.predict([[5,4,1,4.5,1]])
print("Resultado es: " + str(resultado))

#print("Variable interna del modelo")
#print(capa.get_weights())