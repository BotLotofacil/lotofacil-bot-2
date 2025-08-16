import pandas as pd

def converter_csv_lotofacil(entrada: str, saida: str):
    try:
        # Lê o CSV separando por ponto e vírgula (;)
        df = pd.read_csv(entrada, sep=';')

        # Filtra apenas colunas com "Bola" no nome
        colunas_bolas = [col for col in df.columns if 'Bola' in col]
        df_bolas = df[colunas_bolas]

        # Converte os valores para inteiros
        df_bolas = df_bolas.applymap(lambda x: int(str(x).strip()))

        # Salva no formato correto (sem cabeçalho, separado por vírgula)
        df_bolas.to_csv(saida, index=False, header=False)

        print(f"✅ Arquivo convertido com sucesso: {saida}")
    except Exception as e:
        print(f"❌ Erro ao converter arquivo: {e}")

if __name__ == "__main__":
    converter_csv_lotofacil("history.csv", "history_convertido.csv")
