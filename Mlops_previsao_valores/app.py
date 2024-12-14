from flask import Flask, request, jsonify
import pickle

# Carregar o modelo

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber os dados em formato JSON
        data = request.get_json()

        # Validar entrada

        if 'tamanho' not in data:
            return jsonify({'error': 'O campo "tamanho" é obrigatorio'}), 400
        
        tamanho = float(data['tamanho'])

        # Fazer previsão
        preco = model.predict([[tamanho]])

        # Retornar o resultado

        return jsonify({'preco': preco[0]})
    except ValueError:
        return jsonify({'error': 'O campo "tamanho" deve ser numérico!'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)