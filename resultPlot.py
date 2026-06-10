import json
import matplotlib.pyplot as plt
import os

def plot_loss(log_json_path,save_name='loss_fig.png',smoothed_weight=0.85):
    steps = []
    losses = []

    with open(log_json_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            if 'loss' in data:
                step = data.get('step',data.get('iter'))
                loss = data.get('loss')
                steps.append(step)
                losses.append(loss)
    smoothed_losses = []
    last = losses[0]
    for point in losses:
        smoothed_val = last*smoothed_weight + (1-smoothed_weight) * point
        smoothed_losses.append(smoothed_val)
        last = smoothed_val


    plt.figure(figsize=(10,6))
    plt.plot(steps,losses,color='C0',alpha=0.5,label='Raw Loss')
    plt.plot(steps,smoothed_losses,color='C0',linewidth=2,label=f'Smoothed_loss (w={smoothed_weight})')
    plt.title('Training loss curve',fontsize=16)
    plt.xlabel('Iterations',fontsize=14)
    plt.ylabel('Loss',fontsize=14)
    plt.grid(True,linestyle='--',alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_name,dpi=300,bbox_inches='tight')


if __name__ == '__main__':
    # json_file = '/mnt/ht2_nas2/00-model/00-fb/mmseg_dino/work_dirs/20260609_190810/vis_data/20260609_190810.json'
    json_file = '/mnt/ht2_nas2/00-model/00-fb/mmseg_dino/work_dirs/20260610_111831/vis_data/20260610_111831.json'
    plot_loss(json_file,save_name='loss_fig_ft.png')
