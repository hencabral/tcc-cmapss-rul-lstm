# 📘 Notebooks de Treinamento — FD001 (C-MAPSS)

Este diretório contém os notebooks utilizados no TCC para treinar e avaliar os modelos MLP e LSTM no conjunto FD001 do C-MAPSS.

Todos os notebooks já estão configurados para ler os dados diretamente da pasta **CMAPSSData** presente neste repositório.  
Não é preciso modificar caminhos nem criar pastas adicionais.

---

## ▶️ Como executar no Google Colab

1. Abra o notebook no Colab (clicando em *"Open in Colab"*).
2. No painel lateral do Colab, clique em **Arquivos → Upload**.
3. Envie a pasta **CMAPSSData** completa (que já está neste repositório).
4. Execute o notebook inteiro.

---

## ▶️ Como executar localmente (Jupyter / VSCode)

1. Instale as dependências:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib joblib
