import numpy as np 
import tensorflow as tf
import random
from flask import Flask, render_template, request

#entrada = np.array([[10,10,1,10,1],[9,9,1,9,1],[8,8,1,8,1],[7,7,1,7,1],[6,6,1,6,1],[5,5,1,5,1],[4,4,1,4,1],[3,3,1,3,1],[2,2,1,2,1],[1,1,1,1,1], [10, 10, 1, 0, 1]], dtype=float)                          
#salida = np.array([[1,0,0],[0.9, 0.05, 0.05],[0.8, 0.1, 0.1],[0.7, 0.15, 0.15],[0.6, 0.2, 0.2],[0.5, 0.25, 0.25],[0.4, 0.3, 0.3],[0.3, 0.35, 0.35],[0.2, 0.4, 0.4],[0.1, 0.45, 0.45], [0, 0.5, 0.5]], dtype=float) #, dtype=float

entrada = np.array([[10,10,10],[9,9,9],[8,8,8],[7,7,7],
                    [6,6,6],[5,5,5],[4,4,4],[3,3,3],
                    [2,2,2],[1,1,1], [10,10,0],
                    [5,10,6],[7,8,1],[9,5,6],[1,1,7],[3,3,7],[4,4,7]], dtype=float) 
                         
salida  = np.array([[1,0,0],[0.9, 0.05, 0.05],[0.8, 0.1, 0.1],[0.7, 0.15, 0.15],
                    [0.6, 0.2, 0.2],[0.5, 0.25, 0.25],[0.4, 0.3, 0.3],[0.3, 0.35, 0.35],
                    [0.2, 0.4, 0.4],[0.1, 0.45, 0.45], [0, 0.5, 0.5],
                    [0.6,0.3,0.1],[0.2,0.5,0.3],[0.6,0.1,0.3],[0.6,0.2,0.2],[0.6,0.2,0.2],[0.7,0.15,0.15]], dtype=float) #, dtype=float

app = Flask(__name__)

def recomendacion(pas, mtiempo, mexer):
    if(pas > 0.7):
        reco=("¡Felicitaciones! pasaste nivel")
        ref=1
    else:
        if(mtiempo > mexer):
            reco=("No aprobaste...\nAumenta el tiempo de estudio ")
            ref=2
        else:
            reco=("No aprobaste...\nAumenta el número de ejercicios ")
            ref=3
    return (reco, ref)

def RedNeuronal(t, e, nt):
    capa = tf.keras.layers.Dense(units = 3, input_shape = [3], activation='sigmoid')
    modelo = tf.keras.Sequential([capa])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1), #Que tanto ajustar pesos y sesgos
        loss='mean_squared_error',
    )

    print("Comensando entrenamiento")
    historial = modelo.fit(entrada, salida, epochs=500, verbose=False) #Vueltas = 500
    print("Modelo entrenado")

    print("Hagamos una prediccion")
    resultado = modelo.predict([[t, e, nt]])
    resultado2 = resultado.tolist()
    #print("Resultado es: " + str(resultado))
    
    Pasa = resultado[0,0]
    MasT = resultado[0,1]
    MasE = resultado[0,2]

    return (resultado2, Pasa, MasT, MasE)


def check(res, sol):
    if res == sol:
        res = 1
    else:
        res=0
    return res

def tex(Resul1):
    if Resul1 == 1:
        Resul1 ="Correcto"
    else:
        Resul1 ="Incorrecto"
    return Resul1

@app.route('/')
def principal():
    return render_template('index.html')

@app.route('/datos')
def datos():
    return render_template('datos.html')

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')


@app.route('/red', methods = ['POST'])
def red():
    Tiempo=request.form['Tiempo']
    Ejercicios=request.form['Ejercicios']
    NotaTest=request.form['NotaTest']

    (resultado, pas, mtiempo, mexer) = RedNeuronal(float(Tiempo), float(Ejercicios), float(NotaTest))
    (recomen, ref) = recomendacion(float(pas), float(mtiempo), float(mexer))

    return render_template("ver.html", time=Tiempo, exer=Ejercicios, notaT=NotaTest, resul = resultado, pas = pas, mtiempo = mtiempo, mexer = mexer, recomen=recomen, ref=ref)


@app.route('/form', methods = ['POST'])
def form():
    Nombre=request.form['Nombre']
    Res1=request.form['r1']
    Res2=request.form['r2'] 
    Res3=request.form['r3']
    Res4=request.form['r4'] 
    Res5=request.form['r5']
    Res6=request.form['r6']
    Res7=request.form['r7'] 
    Res8=request.form['r8']
    Res9=request.form['r9'] 
    Res10=request.form['r10']


    Resul1 = check(Res1, '1')
    Resul2 = check(Res2, '2')
    Resul3 = check(Res3, '3')
    Resul4 = check(Res4, '1')
    Resul5 = check(Res5, '1')
    Resul6 = check(Res6, '3')
    Resul7 = check(Res7, '2')
    Resul8 = check(Res8, '1')
    Resul9 = check(Res9, '1')
    Resul10 = check(Res10, '1')

    Resultado=(Resul1 + Resul2 + Resul3 + Resul4 + Resul5 + Resul6 + Resul7 + Resul8 + Resul9 + Resul10)

    if Resultado < 7:
        img=5
    else:
        img =4
    Resul1 = tex(Resul1)
    Resul2 = tex(Resul2)
    Resul3 = tex(Resul3)
    Resul4 = tex(Resul4)
    Resul5 = tex(Resul5)
    Resul6 = tex(Resul6)
    Resul7 = tex(Resul7)
    Resul8 = tex(Resul8)
    Resul9 = tex(Resul9)
    Resul10 = tex(Resul10)

    Tiempo = random.randint(4, 10)
    Num_ejercicios = random.randint(4,10)


    (resultado, pas, mtiempo, mexer) = RedNeuronal(float(Tiempo), float(Num_ejercicios), float(Resultado))
    (recomen, ref) = recomendacion(float(pas), float(mtiempo), float(mexer))


    return render_template("ver2.html", imagen=img, Resultado_final = Resultado, Nom = Nombre, Resultado1=Resul1, Resultado2=Resul2, 
                            Resultado3=Resul3, Resultado4=Resul4, Resultado5=Resul5,
                            Resultado6=Resul6, Resultado7=Resul7, Resultado8=Resul8,
                            Resultado9=Resul9, Resultado10=Resul10,
                            time=Tiempo, exer=Num_ejercicios, notaT=Resultado, resul = resultado, pas = pas, mtiempo = mtiempo, mexer = mexer,
                            recomen=recomen, ref=ref )


if __name__ == '__main__':
    app.run(debug=True, port=5017)

#python .\index.py