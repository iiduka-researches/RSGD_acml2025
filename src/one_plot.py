import pandas as pd
import matplotlib.pyplot as plt
import optimizers

info = 'grad_norm' # grad_norm or loss
d_output = 'show' # show or save

dataset_name = 'jester'
bs_name = 'PolyGrowthBS'
lr_name = 'WarmUpLR'
lr_dacaypart = 'PolyLR' # only WarmUpLR
list_1 = [0.5, 0.1, 0.05, 0.01, 0.005]
list_2 = [3, 8]
list_3 = [0.5, 0.05, 0.005]
list_4 = []


if lr_name == 'WarmUpLR':
    if not (bs_name == 'ExpoGrowthBS' or bs_name == 'PolyGrowthBS'):
        raise NameError(f'The batch size type does not found: {bs_name}')

base_dir = f'results/official_results/{dataset_name}/{bs_name}/{lr_name}'

if not lr_name == 'WarmUpLR':
    for i in list_1:
        data = pd.read_pickle(f'{base_dir}/RSGD-{bs_name}{lr_name}-initlr{i}.pkl')
        plt.plot(data[info], label=f'{bs_name}{lr_name}-initlr{i}')
else:
    for i in list_2:
        for j in list_3:
            data = pd.read_pickle(f'{base_dir}/lr_up:{i}/{lr_dacaypart}/RSGD-{bs_name}Expo{lr_name}and{lr_dacaypart}-lrmax{j}.pkl')
            plt.plot(data[info], label=f'{bs_name}(dc){lr_dacaypart}(up){i}-lrmax{j}')
            
plt.xlabel('Iteration')
plt.ylabel(info)
plt.title(f'{info} Over Iterations')
plt.legend()
plt.grid()
plt.tight_layout()

if d_output == 'save':
    if lr_name == 'WarmUpLR':
        plt.savefig(f'{dataset_name}-{bs_name}-{lr_name}-(dc){lr_dacaypart}.png')
    else:
        plt.savefig(f'{dataset_name}-{bs_name}-{lr_name}.png')
elif d_output == 'show':
    plt.show()