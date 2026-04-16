import torch
from models.dynamics.gptdynamics import WorldModel, WorldModelConfig

def test_sanity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Configuração idêntica ao seu treino
    config = WorldModelConfig(
        vocab_size=512,
        n_embd=256,
        n_head=8,
        n_layer=4,
        tokens_per_frame=64,
        frames_per_seq=20
    )
    
    print("🤖 Inicializando modelo para teste...")
    model = WorldModel(config).to(device)
    model.eval()

    # 2. Criar dados "fake" com as dimensões que o seu DataLoader entrega
    # Batch=1, Frame=1, Tokens=64
    fake_img_tokens = torch.randint(0, 512, (1, 1, 64)).to(device)
    # Batch=1, Seq=20, Ações=3
    fake_actions = torch.randn(1, 20, 3).to(device)

    print("✨ Testando geração de 20 frames...")
    try:
        # Tenta gerar 19 passos (para totalizar 20 frames com o inicial)
        tokens_imaginados = model.generate(fake_img_tokens, fake_actions, max_steps=19)
        
        print(f"✅ Sucesso! Formato de saída: {tokens_imaginados.shape}")
        if tokens_imaginados.shape == (1, 20, 64):
            print("💎 As dimensões estão perfeitas: [Batch=1, Time=20, Tokens=64]")
        else:
            print(f"⚠️ Formato inesperado: {tokens_imaginados.shape}")
            
    except Exception as e:
        print(f"❌ O erro ainda persiste:\n{e}")

if __name__ == "__main__":
    test_sanity()