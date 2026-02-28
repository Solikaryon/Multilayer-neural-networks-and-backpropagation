import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Luis Fernando Monjaraz Briseño

# Funciones de activación

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return math.tanh(x)

def dtanh(y):
    return 1 - y * y


# Clase MLP

class MLP:
    def __init__(self, n_input, n_hidden, n_output, activation='sigmoid', lr=0.5, weight_init='random_0_1.5'):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
        self.activation = activation
        self.weight_init = weight_init
        
        # Seleccionar función de activación
        if activation == 'sigmoid':
            self.act_func = sigmoid
            self.dact_func = dsigmoid
        elif activation == 'tanh':
            self.act_func = tanh
            self.dact_func = dtanh

        # Inicializar pesos según especificación
        self.w_input_hidden = self._initialize_weights(n_input, n_hidden)
        self.bias_hidden = self._initialize_weights(1, n_hidden)[0]
        
        self.w_hidden_output = self._initialize_weights(n_hidden, n_output)
        self.bias_output = self._initialize_weights(1, n_output)[0]
        
        self.mse_history = []

    def _initialize_weights(self, rows, cols):
        """Inicializa pesos según el método especificado"""
        if self.weight_init == 'random_0_1.5':
            # Pesos aleatorios entre 0 y 1.5
            return [[random.uniform(0, 1.5) for _ in range(cols)] for _ in range(rows)]
        elif self.weight_init == 'binary_0_1.5':
            # Pesos que son solo 0 o 1.5
            return [[random.choice([0, 1.5]) for _ in range(cols)] for _ in range(rows)]
        else:
            return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

    def forward(self, inputs):
        # Capa oculta
        self.hidden = []
        for j in range(self.n_hidden):
            suma = sum(inputs[i] * self.w_input_hidden[i][j] for i in range(self.n_input)) + self.bias_hidden[j]
            self.hidden.append(self.act_func(suma))

        # Capa de salida
        self.output = []
        for k in range(self.n_output):
            suma = sum(self.hidden[j] * self.w_hidden_output[j][k] for j in range(self.n_hidden)) + self.bias_output[k]
            self.output.append(self.act_func(suma))

        return self.output

    def backpropagation(self, inputs, targets):
        # Calcular error de salida
        output_deltas = [0] * self.n_output
        for k in range(self.n_output):
            error = targets[k] - self.output[k]
            output_deltas[k] = error * self.dact_func(self.output[k])

        # Calcular error de la capa oculta
        hidden_deltas = [0] * self.n_hidden
        for j in range(self.n_hidden):
            error = sum(output_deltas[k] * self.w_hidden_output[j][k] for k in range(self.n_output))
            hidden_deltas[j] = error * self.dact_func(self.hidden[j])

        # Actualizar pesos oculta -> salida
        for j in range(self.n_hidden):
            for k in range(self.n_output):
                self.w_hidden_output[j][k] += self.lr * output_deltas[k] * self.hidden[j]
        for k in range(self.n_output):
            self.bias_output[k] += self.lr * output_deltas[k]

        # Actualizar pesos entrada -> oculta
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                self.w_input_hidden[i][j] += self.lr * hidden_deltas[j] * inputs[i]
        for j in range(self.n_hidden):
            self.bias_hidden[j] += self.lr * hidden_deltas[j]

        # Error cuadrático medio
        mse = sum((targets[k] - self.output[k])**2 for k in range(self.n_output)) / self.n_output
        return mse

    def train(self, data, epochs=10000):
        self.mse_history = []
        for epoch in range(epochs):
            mse = 0
            for inputs, targets in data:
                self.forward(inputs)
                mse += self.backpropagation(inputs, targets)
            mse /= len(data)
            self.mse_history.append(mse)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, MSE = {mse:.6f}")

    def predict(self, inputs):
        return self.forward(inputs)

# PRÁCTICA 1: APLICACIÓN DEL MLP CON BACKPROPAGATION (XOR/¬XOR)

def practica_1_xor():
    print("=" * 80)
    print("PRÁCTICA 1: APLICACIÓN DEL MLP CON BACKPROPAGATION")
    print("Problema: XOR/¬XOR")
    print("=" * 80)
    
    # Datos: [x1, x2] -> [XOR, ¬XOR]
    data = [
        ([0,0], [0,1]),
        ([0,1], [1,0]),
        ([1,0], [1,0]),
        ([1,1], [0,1]),
    ]

    # Configuración específica del problema: 2-2-2 neuronas
    print("Configuración de la red: 2 neuronas entrada, 2 neuronas ocultas, 2 neuronas salida")
    print("Función de activación: Sigmoide")
    print("Inicialización de pesos: Valores aleatorios entre 0 y 1.5")
    print("-" * 60)
    
    mlp = MLP(2, 2, 2, activation='sigmoid', lr=0.5, weight_init='random_0_1.5')
    
    print("Pesos iniciales:")
    print(f"Entrada->Oculta: {np.round(mlp.w_input_hidden, 4)}")
    print(f"Bias oculta: {np.round(mlp.bias_hidden, 4)}")
    print(f"Oculta->Salida: {np.round(mlp.w_hidden_output, 4)}")
    print(f"Bias salida: {np.round(mlp.bias_output, 4)}")
    
    print("\nEntrenando la red...")
    mlp.train(data, epochs=10000)

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES - PRÁCTICA 1")
    print("=" * 60)
    
    print("Pesos finales:")
    print(f"Entrada->Oculta: {np.round(mlp.w_input_hidden, 4)}")
    print(f"Bias oculta: {np.round(mlp.bias_hidden, 4)}")
    print(f"Oculta->Salida: {np.round(mlp.w_hidden_output, 4)}")
    print(f"Bias salida: {np.round(mlp.bias_output, 4)}")
    
    print("\nVerificación de salidas:")
    print("-" * 50)
    for inputs, target in data:
        output = mlp.predict(inputs)
        rounded_output = [round(o, 4) for o in output]
        print(f"Entrada: {inputs}, Esperado: {target}, Salida: {rounded_output}")

    # Graficar MSE
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp.mse_history)
    plt.title('PRÁCTICA 1: Evolución del MSE - Problema XOR/¬XOR')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    text = "PRÁCTICA 1: RESULTADOS XOR/¬XOR\n\n"
    text += "Configuración:\n"
    text += "- Red: 2-2-2 neuronas\n"
    text += "- Activación: Sigmoide\n"
    text += "- Pesos iniciales: 0 a 1.5\n\n"
    text += "Pesos finales:\n"
    text += f"Entrada->Oculta:\n{np.round(mlp.w_input_hidden, 4)}\n"
    text += f"Bias oculta: {np.round(mlp.bias_hidden, 4)}\n"
    text += f"Oculta->Salida:\n{np.round(mlp.w_hidden_output, 4)}\n"
    text += f"Bias salida: {np.round(mlp.bias_output, 4)}"
    plt.text(0.1, 0.5, text, fontfamily='monospace', verticalalignment='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# PRÁCTICA 2: APROXIMACIÓN DE FUNCIONES CON MLP

def practica_2_aproximacion_funciones():
    print("\n" + "=" * 80)
    print("PRÁCTICA 2: APROXIMACIÓN DE FUNCIONES CON MLP")
    print("=" * 80)
    
    # Función 1: f(x) = 2cos(2x) - sen(x) con 0 ≤ x ≤ 2π
    def f1(x):
        return 2 * math.cos(2*x) - math.sin(x)
    
    # Función 2: f(x) = 2cos(2x) - sen(x) + 0.9 * exp(sin(3x)) con -2π ≤ x ≤ 2π
    def f2(x):
        return 2 * math.cos(2*x) - math.sin(x) + 0.9 * math.exp(math.sin(3*x))
    
    # Configuraciones a probar
    configs = [
        (5, 0.1, 'sigmoid'),
        (10, 0.1, 'sigmoid'),
        (15, 0.1, 'sigmoid'),
        (5, 0.3, 'tanh'),
        (10, 0.3, 'tanh'),
        (15, 0.3, 'tanh'),
    ]
    
    # Probar ambas funciones
    functions = [
        (f1, "f(x) = 2cos(2x) - sen(x)", 0, 2*math.pi, 100, "Función 1"),
        (f2, "f(x) = 2cos(2x) - sen(x) + 0.9*exp(sin(3x))", -2*math.pi, 2*math.pi, 200, "Función 2")
    ]
    
    for func, func_name, x_min, x_max, n_points, func_id in functions:
        print(f"\n{'='*60}")
        print(f"PRÁCTICA 2 - {func_id}")
        print(f"Aproximando: {func_name}")
        print(f"Dominio: [{x_min:.2f}, {x_max:.2f}]")
        print(f"{'='*60}")
        
        # Generar datos
        x_train = [x_min + (x_max - x_min) * i / n_points for i in range(n_points)]
        y_train = [func(x) for x in x_train]
        
        # Normalizar datos (entrada y salida)
        x_min_val, x_max_val = min(x_train), max(x_train)
        y_min_val, y_max_val = min(y_train), max(y_train)
        
        x_norm = [(x - x_min_val) / (x_max_val - x_min_val) for x in x_train]
        y_norm = [(y - y_min_val) / (y_max_val - y_min_val) for y in y_train]
        
        data = [([x_norm[i]], [y_norm[i]]) for i in range(n_points)]
        
        best_mse = float('inf')
        best_config = None
        
        for n_hidden, lr, activation in configs:
            print(f"\n--- Probando: {n_hidden} neuronas ocultas, lr={lr}, activación={activation} ---")
            
            mlp = MLP(1, n_hidden, 1, activation=activation, lr=lr)
            mlp.train(data, epochs=5000)
            
            # Predecir
            predictions = []
            for x in x_norm:
                pred_norm = mlp.predict([x])[0]
                pred = pred_norm * (y_max_val - y_min_val) + y_min_val
                predictions.append(pred)
            
            # Calcular error
            mse_final = sum((predictions[i] - y_train[i])**2 for i in range(n_points)) / n_points
            
            if mse_final < best_mse:
                best_mse = mse_final
                best_config = (n_hidden, lr, activation, mlp, predictions)
            
            print(f"MSE final: {mse_final:.6f}")
            
            # Graficar resultados individuales
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(x_train, y_train, 'b-', label='Función original', linewidth=2)
            plt.plot(x_train, predictions, 'r--', label='MLP aproximación', linewidth=2)
            plt.title(f'PRÁCTICA 2 - {func_id}\n{activation}, {n_hidden} neuronas, lr={lr}\nMSE: {mse_final:.6f}')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(mlp.mse_history)
            plt.title('Evolución del MSE durante entrenamiento')
            plt.xlabel('Época')
            plt.ylabel('MSE')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            error = [predictions[i] - y_train[i] for i in range(n_points)]
            plt.plot(x_train, error, 'g-', label='Error', alpha=0.7)
            plt.title('Error de aproximación')
            plt.xlabel('x')
            plt.ylabel('Error')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        # Mostrar mejor configuración
        if best_config:
            n_hidden, lr, activation, best_mlp, best_predictions = best_config
            print(f"\n{'='*60}")
            print(f"MEJOR CONFIGURACIÓN PARA {func_id}:")
            print(f"Neuronas ocultas: {n_hidden}, Learning rate: {lr}, Activación: {activation}")
            print(f"MSE: {best_mse:.6f}")
            print(f"{'='*60}")
            
            # Gráfica final con mejor configuración
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(x_train, y_train, 'b-', label='Función original', linewidth=3)
            plt.plot(x_train, best_predictions, 'r--', label='MLP aproximación', linewidth=2)
            plt.title(f'PRÁCTICA 2 - MEJOR APROXIMACIÓN\n{func_name}\n{activation}, {n_hidden} neuronas, lr={lr}', fontsize=12)
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(best_mlp.mse_history)
            plt.title('Evolución del MSE (Mejor configuración)')
            plt.xlabel('Época')
            plt.ylabel('MSE')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            error = [best_predictions[i] - y_train[i] for i in range(n_points)]
            plt.plot(x_train, error, 'g-', label='Error', alpha=0.7)
            plt.title('Error de aproximación')
            plt.xlabel('x')
            plt.ylabel('Error')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.axis('off')
            text = f"MEJOR CONFIGURACIÓN:\n\n"
            text += f"Función: {func_id}\n"
            text += f"Neuronas ocultas: {n_hidden}\n"
            text += f"Learning rate: {lr}\n"
            text += f"Activación: {activation}\n"
            text += f"MSE final: {best_mse:.6f}\n\n"
            text += f"Pesos finales:\n"
            text += f"Entrada->Oculta:\n{np.round(best_mlp.w_input_hidden, 4)}\n"
            text += f"Bias oculta: {np.round(best_mlp.bias_hidden, 4)}\n"
            text += f"Oculta->Salida:\n{np.round(best_mlp.w_hidden_output, 4)}\n"
            text += f"Bias salida: {np.round(best_mlp.bias_output, 4)}"
            plt.text(0.1, 0.5, text, fontfamily='monospace', verticalalignment='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()

# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    print("IMPLEMENTACIÓN DE MLP CON BACKPROPAGATION")
    print("=" * 80)
    
    # Ejecutar Práctica 1
    practica_1_xor()
    
    # Ejecutar Práctica 2
    practica_2_aproximacion_funciones()
    
    print("\n" + "=" * 80)
    print("TODAS LAS PRÁCTICAS COMPLETADAS EXITOSAMENTE")
    print("=" * 80)