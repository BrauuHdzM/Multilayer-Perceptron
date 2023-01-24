from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib

### Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
	
X_train = df.drop(['label'],axis=1).values   
y_train = df['label'].values

df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
	
X_test = df.drop(['label'],axis=1).values   
y_test = df['label'].values

### Asignar parametros al modelo

parameters = {'random_state':[0], 'hidden_layer_sizes':np.arange(10, 12), 'activation':['relu', 'logistic'], 'learning_rate':['constant', 'invscaling'], 'alpha': [1.0, 0.1, 0.01], 'max_iter': [ 600, 800, 1000 ]}

### Training
clf = MLPClassifier(random_state=0)

cv = GridSearchCV(clf, parameters, verbose=3)
cv.fit(X_train, y_train)

### Guardar GridSearch 
# joblib.dump(cv, 'MLPClassifier.pkl')


### Cargar GridSearch
# cv = joblib.load("MLPClassifier.pkl")

### Imprimir resultados
df = pd.DataFrame(cv.cv_results_)
print (df)
df.to_csv('cv_results.csv')
print (cv.best_score_)
print (cv.best_params_)

### Testing
y_pred = cv.predict(X_test)

### Evaluacion del modelo
print (classification_report(y_test, y_pred))

### Matriz de confusion
cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

### Impresion de predicciones erroneas
# errors = []
# for i in range(len(y_test)):
#     if (y_test[i] != y_pred[i]):
#         errors.append(i)

count = 0
for i in range(201):
    fig, axarr = plt.subplots(1,4) 
    for j in range(4):
        if (count<803):
            image = np.reshape(X_test[errors[count]], (28, 28))
            axarr[j].set_title('y: '+str(y_test[errors[count]])+'  y^: '+str(y_pred[errors[count]]))
            axarr[j].imshow(image)
        count+=1
    # plt.savefig('errors_'+str(i)+'.png') ## Guardar imagenes
    plt.show() ## Mostrar errores de prediccion

 



	
	
	
	
	
	
	
	
			
