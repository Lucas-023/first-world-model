import os
import time
from huggingface_hub import HfApi, upload_folder

# ==========================================
# ⚙️ CONFIGURAÇÕES PARA O DATASET
# ==========================================
HF_TOKEN = "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"
REPO_ID = "Lucas-023/dataset" # Recomendo um repo separado para o dataset

# A pasta onde estão as suas imagens 64x64 e o metadata.csv
LOCAL_FOLDER_PATH = "dataset_jogo" 

# No Hugging Face, os datasets costumam ficar no tipo "dataset"
REPO_TYPE = "dataset"

# 600 segundos = 10 minutos
INTERVALO_SEGUNDOS = 600 
# ==========================================

def upload_dataset():
    if not os.path.isdir(LOCAL_FOLDER_PATH):
        print(f"⚠️ Pasta {LOCAL_FOLDER_PATH} não encontrada.")
        return False

    print(f"🚀 Sincronizando dataset com {REPO_ID}...")
    try:
        # A função upload_folder é mais eficiente para pastas
        upload_folder(
            folder_path=LOCAL_FOLDER_PATH,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            commit_message="Sincronização automática do dataset"
        )
        print("✅ Sincronização concluída com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro na sincronização: {e}")
        return False

if __name__ == "__main__":
    print(f"🔄 Monitor de Dataset iniciado. Sincronizando a cada {INTERVALO_SEGUNDOS}s...")
    
    while True:
        if os.path.exists(LOCAL_FOLDER_PATH):
            print("\n" + "="*40)
            print(f"🕒 Verificando novos dados na pasta...")
            upload_dataset()
        else:
            print(f"🔍 Aguardando a criação da pasta {LOCAL_FOLDER_PATH}...")
            
        time.sleep(INTERVALO_SEGUNDOS)