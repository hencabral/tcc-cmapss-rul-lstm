# 🔧 RUL Predictor – LSTM Residual (FD001 - C-MAPSS)

Este repositório contém o protótipo funcional desenvolvido para o TCC, capaz de estimar a **Vida Útil Remanescente (RUL)** de motores turbofan do conjunto **NASA C-MAPSS (FD001)** utilizando o **modelo LSTM Residual**.

A aplicação é executada localmente via **Streamlit**, permitindo o envio de arquivos de sensores e a visualização das predições de forma interativa.

---

## 📘 Contexto do Projeto

Este trabalho investiga técnicas de aprendizado profundo para estimar a **vida útil remanescente (RUL)** de sistemas aeronáuticos simulados. O subset **FD001** do dataset **NASA C-MAPSS** foi utilizado por sua estrutura de 1 condição operacional e 1 regime de falha.  
Além do desenvolvimento do modelo LSTM, também foi criado um **protótipo web** para demonstrar sua aplicação prática, permitindo testar novos arquivos e analisar o comportamento dos sensores ao longo do ciclo de vida dos motores.

---

## 🧰 Tecnologias Utilizadas

- Python 3  
- TensorFlow / Keras  
- NumPy  
- pandas  
- Streamlit  
- Matplotlib  
- scikit-learn  

---

## 🚀 Como Executar o Protótipo

### 1. Instale as dependências

No terminal:

```bash
pip install -r app/requirements.txt


```bash
streamlit run app/app.py