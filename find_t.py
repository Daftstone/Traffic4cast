import pandas as pd
import numpy as np
import collections


def count_dou(s):
    cc = 0
    for ss in s:
        if (ss == ','):
            cc += 1
    return cc


city = 'melbourne'

daily_counts = pd.read_parquet(f"data/loop_counter/{city}/counters_daily_by_node.parquet")

test_data = pd.read_parquet(f"data/test/{city}/input/counters_test.parquet")
# train_data = pd.read_parquet(f"data/train/{city}/input/counters_2020-06-01.parquet")
#
# print(train_data)
# # exit(0)

info_df = pd.DataFrame(columns=['test_idx', 'day', 't'])

for i in range(100):
    maps = collections.defaultdict(lambda: 0)
    cur_test_data = test_data[test_data['test_idx'] == i]
    count = []
    cur_info = ""
    del_maps = collections.defaultdict(lambda: 0)
    for _, row in cur_test_data.iterrows():
        volumes = ", ".join([str(s) for s in row['volumes_1h']])
        if (volumes == 'nan, nan, nan, nan'):
            continue
        filter_data = daily_counts[daily_counts['node_id'] == row['node_id']]
        for _, row1 in filter_data.iterrows():
            if (del_maps["%s" % (row1['day'])] == 1):
                continue
            volumes1 = ", ".join([str(s) for s in row1['volume']])
            if (volumes in volumes1):
                idx = volumes1.find(volumes)
                t = count_dou(volumes1[:idx]) + 4
                maps["%s_%s" % (row1['day'], t)] += 1
            else:
                del_maps["%s" % (row1['day'])] = 1
        #         if (maps["%s_%s" % (row1['day'], t)] >= len(cur_test_data) - 100):
        #             cur_info = "%s_%s" % (row1['day'], t)
        #             break
        # if (len(cur_info) > 0):
        #     break
    # maps1 = sorted(maps.items(), key=lambda x: x[1], reverse=True)[:20]
    cur_info = max(maps, key=maps.get)
    ss = cur_info.split("_")
    day = ss[0]
    t = int(ss[1])
    print(day, t, volumes)
    info_df.loc[i] = [i, day, t]
print(len(info_df))
info_df.to_csv(f"data/{city}_test_info.csv", index=False)
