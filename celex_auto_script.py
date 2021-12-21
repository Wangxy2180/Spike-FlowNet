# nohup python server_acc.py 2>&1 > log.log &
# nohup python server_acc.py 2>&1 | tee jintian.log &

import os
import time
import re
import copy


def key_info_extract(files):
    AEE_line = files[-2]
    act_line = files[-4]
    # 这里应该使用懒惰匹配
    act = re.findall(r'tensor\((.*), device', act_line)
    AEE = re.findall(r'Mean AEE: (.*), sum AEE:', AEE_line)

    return eval(AEE[0]), eval(act[0])


def list_to_str(data_list):
    data_str = []
    for i in data_list:
        data_str.append(str(i))
    data_str = ','.join(data_str)
    return data_str


def save_data(file_path, mem_thr_str, result):
    with open(file_path, 'w') as f:
        f.write(mem_thr_str + '\n')
        for k in result:
            f.write(k + ',')
            ret_str = list_to_str(result[k])
            f.write(ret_str + '\n')


def main():
    # 0.5~1.5
    # mem_thr = [round(0.5 + i * 0.05, 2) for i in range(21)]
    # mem_thr = [0.75,  1, 1.1] #for test
    mem_thr = [1]  # for test
    # print(mem_thr)
    # datasets = ['walk', 'jiayou', 'polygon_camerastatic', 'polygon_cameradynamic', 'polygon_multi_cameradynamic',
    #             'slider_rotation', '6dof_sort']
    datasets = ['spin_fast', 'spin_mid', 'spin_slow', 'spin_slow_mid', 'spin_static', 'spin_vf', 'spin_vs']
    # datasets = ['jiayou', 'walk']  # for test
    empty_val = [[] for i in range(len(datasets))]

    # 这问题也太坑爹了，这里的zip居然是浅copy直接把empty_val的地址拿过来了
    act_result = dict(zip(datasets, copy.deepcopy(empty_val)))
    AEE_result = dict(zip(datasets, copy.deepcopy(empty_val)))
    # files = []
    pre_train_ckp = r'/home/wxy/ec/SNNCode/Spike-FlowNet/pretrain/celex_10000epoch.pth.tar'
    for dataset in datasets:
        for thr in mem_thr:
            exec_cmd = f'/home/wxy/anaconda3/envs/spikeflownet/bin/python /home/wxy/ec/SNNCode/Spike-FlowNet/celex_validCelexData.py --test-env={dataset} --mem-thr {thr} --evaluate --pretrained={pre_train_ckp} 2>&1 | tee /home/wxy/ec/eva/{dataset}_{thr}.txt'
            # os.system(exec_cmd)
            files = os.popen(exec_cmd).readlines()
            AEE, act = key_info_extract(files)
            act_result[dataset].append(act)
            AEE_result[dataset].append(AEE)
            print(exec_cmd)

    # 开始保存数据
    mem_thr_str = list_to_str(mem_thr)

    save_data('mean_AEE.txt', mem_thr_str, AEE_result)
    save_data('mean_act.txt', mem_thr_str, act_result)

    # with open('mean_AEE.txt', 'w') as f:
    #     f.write(mem_thr_str + '\n')
    #     for k in AEE_result:
    #         f.write(k + ': ')
    #         AEE_str = list_to_str(AEE_result[k])
    #         f.write(AEE_str + '\n')
    #
    # with open('mean_act.txt', 'w') as f:
    #     f.write(mem_thr_str + '\n')
    #     f.write('hello')

    print('done!')


if __name__ == '__main__':
    main()
    # os.system('sleep 5')
    print('done')
    # run_cnt=[6,12,6]
