import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

SAVE_DIR = "dataset_carracing"

def examine_data(episode_id=0):
    filename = os.path.join(SAVE_DIR, f"episode_{episode_id:04d}.npz")
    
    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado!")
        return

    # Carrega os dados
    data = np.load(filename)
    frames = data['obs']
    actions = data['actions']
    
    print(f"Examinando {filename}")
    print(f"Total de frames: {frames.shape[0]}")
    print(f"Shape das imagens: {frames.shape[1:]} (deve ser 64, 64, 3)")
    print(f"Shape das ações: {actions.shape[1:]} (deve ser 3,)")
    
    # Exibe as métricas da primeira ação
    print(f"Ação inicial - Volante: {actions[0][0]:.2f}, Acel: {actions[0][1]:.2f}, Freio: {actions[0][2]:.2f}")

    # Configura o player de vídeo com Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    
    # Renderiza o primeiro frame
    img_display = ax.imshow(frames[0])
    
    # Função de atualização para a animação
    def update(frame_idx):
        img_display.set_array(frames[frame_idx])
        # Mostra os comandos no título do gráfico para conferência
        act = actions[frame_idx]
        ax.set_title(f"Frame {frame_idx} | Vol: {act[0]:.2f} | Acel: {act[1]:.2f} | Fr: {act[2]:.2f}")
        return [img_display]

    # Cria a animação (interval=50 significa 50 milissegundos entre frames, aprox 20 FPS)
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Escolha o ID do episódio que deseja examinar
    examine_data(episode_id=0)