from flask import Flask
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from joblib import dump, load
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, send_from_directory

app = Flask(__name__,static_url_path='/static')
@app.route('/')
def inicio():
  return 'Heart Attack Analysis - Serviço de inferência para chances de ataque cardíaco', 200


from flask import request
@app.route('/model', methods=['POST']) #Block GET requests
def model():
  if request.method == 'POST':

    # Buscando o modelo disponível no s3
    s3_client = boto3.client(service_name='s3')
    s3_client.download_file("ygor-bucket", "heart_attack_model.joblib", "./heart_attack_model.joblib")

    # Com download feito carregamos o modelo para o código
    modelo = load('./heart_attack_model.joblib')

    # Coletando dados do formulário
    dados_cliente = pd.DataFrame({
    'age': [request.form['age']],
    'sex': [request.form['sex']],
    'cp': [request.form['cp']],
    'trtbps': [request.form['trtbps']],
    'chol': [request.form['chol']],
    'fbs': [request.form['fbs']],
    'restecg': [request.form['restecg']],
    'thalachh': [request.form['thalachh']],
    'exng': [request.form['exng']],
    'oldpeak': [request.form['oldpeak']],
    'slp': [request.form['slp']],
    'caa': [request.form['caa']],
    'thall': [request.form['thall']]
    })

    # Gera a inferência
    nova_predicao = modelo.predict(dados_cliente)
    probabilidades_novo_dado = modelo.predict_proba(dados_cliente) * 100

    # Montando apresentação gráfica
    labels = ['Classe 0', 'Classe 1']
    sizes = [probabilidades_novo_dado[0, 0], probabilidades_novo_dado[0, 1]]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    ax.text(0.5, 1.1, f"Resultado da inferência: {nova_predicao}", ha='center', va='center', transform=ax.transAxes)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    resposta = f"<img src='data:image/png;base64,{plot_url}' alt='Probabilidades do Modelo'>"

    plt.close()

    return resposta

@app.route('/model-json', methods=['POST']) #Block GET requests
def model_json():
  request_data = request.get_json()

  # Buscando o modelo disponível no s3
  s3_client = boto3.client(service_name='s3')
  s3_client.download_file("ygor-bucket", "heart_attack_model.joblib", "./heart_attack_model.joblib")

  # Com download feito carregamos o modelo para o código
  modelo = load('./heart_attack_model.joblib')

  # Coletando dados do formulário
  dados_cliente = pd.DataFrame({
    'age': [request_data['age']],
    'sex': [request_data['sex']],
    'cp': [request_data['cp']],
    'trtbps': [request_data['trtbps']],
    'chol': [request_data['chol']],
    'fbs': [request_data['fbs']],
    'restecg': [request_data['restecg']],
    'thalachh': [request_data['thalachh']],
    'exng': [request_data['exng']],
    'oldpeak': [request_data['oldpeak']],
    'slp': [request_data['slp']],
    'caa': [request_data['caa']],
    'thall': [request_data['thall']]
  })

  # Gera a inferência
  nova_predicao = modelo.predict(dados_cliente)

  probabilidades_novo_dado = modelo.predict_proba(dados_cliente) * 100

  resposta = (
      f"Resultado da inferência: {nova_predicao}\n"
      f"Probabilidades associadas a cada classe:\n"
      f"Classe 0: {probabilidades_novo_dado[0, 0]:.2f}%\n"
      f"Classe 1: {probabilidades_novo_dado[0, 1]:.2f}%"
  )

  return resposta


@app.route('/model-results')
def model_results():
    # Buscando as imagens com os resultados armazenados
    s3_client = boto3.client(service_name='s3')
    
    # Baixando as imagens
    s3_client.download_file("ygor-bucket", "matriz_corr.png", "./static/matriz_corr.png")
    s3_client.download_file("ygor-bucket", "matriz_conf.png", "./static/matriz_conf.png")
    s3_client.download_file("ygor-bucket", "resultados_gerais.png", "./static/resultados_gerais.png")

    # Renderizando a página HTML com as imagens
    return send_from_directory('static', 'model-results.html')


@app.route('/model-about')
def model_about():
    return send_from_directory('static', 'about.html')


@app.route('/model-form')
def model_form():
    return send_from_directory('static', 'form.html')



if __name__ == "__main__":
  debug = True # com essa opção como True, ao salvar, o "site" recarrega automaticamente."
  app.run(host='0.0.0.0', port=8888, debug=debug)
