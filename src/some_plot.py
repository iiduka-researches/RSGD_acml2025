import pandas as pd
import matplotlib.pyplot as plt
import optimizers
import argparse

info = 'loss' # grad_norm or loss
d_output = 'show' # show or save
min_conv = 'yes' # yes or no

dataset_name = 'jester'
bs_name_list = ['ConstantBS', 'ExpoGrowthBS', 'PolyGrowthBS']
lr_name_list = ['ConstantLR', 'DiminishingLR', 'CosAnnealLR', 'PolyLR', 'WarmUpLR']
lr_dacaypart_list = ['ConstantLR', 'DiminishingLR','CosAnnealLR', 'PolyLR'] # only WarmUpLR
list_1 = [0.5, 0.1, 0.05, 0.01, 0.005]
list_2 = [3, 8]
list_3 = [0.5, 0.05, 0.005]
list_4 = []

for bs_name in bs_name_list:
    for lr_name in lr_name_list:
        if min_conv == 'yes':
            base_dir = f'results/min_converted_results/{dataset_name}/{bs_name}/{lr_name}'
        elif min_conv == 'no':
            base_dir = f'results/official_results/{dataset_name}/{bs_name}/{lr_name}'
            
        if not lr_name == 'WarmUpLR':
            for i in list_1:
                data = pd.read_pickle(f'{base_dir}/RSGD-{bs_name}{lr_name}-initlr{i}.pkl')
                plt.plot(data[info], label=f'{bs_name}{lr_name}-initlr{i}')
        else:
            for lr_dacaypart in lr_dacaypart_list:
                for i in list_2:
                    for j in list_3:
                        data = pd.read_pickle(f'{base_dir}/lr_up:{i}/{lr_dacaypart}/RSGD-{bs_name}Expo{lr_name}and{lr_dacaypart}-lrmax{j}.pkl')
                        plt.plot(data[info], label=f'{bs_name}{lr_name}(dc){lr_dacaypart}(up){i}-lrmax{j}')
            
plt.yscale('log', base=10)

plt.xlabel('Iteration')
plt.ylabel(info)
#plt.title(f'{info} Over Iterations')
plt.legend()
plt.grid()

if d_output == 'save':
    if lr_name_list[0] == 'WarmUpLR':
        plt.savefig(f'{dataset_name}/wupLR/{info}-{lr_name_list[0]}{lr_dacaypart_list[0]}-lrmax{list_3[0]}-lrup{list_2[0]}.pdf', dpi=800, bbox_inches="tight")
    else:
        plt.savefig(f'{dataset_name}/{info}-{lr_name_list[0]}-lrmax{list_1[0]}.pdf', dpi=800, bbox_inches="tight")
elif d_output == 'show':
    plt.show()