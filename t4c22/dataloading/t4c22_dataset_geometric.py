#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pandas as pd
from functools import partial
from typing import Optional

import torch
import torch_geometric

from t4c22.dataloading.road_graph_mapping import TorchRoadGraphMapping
from t4c22.t4c22_config import cc_dates
from t4c22.t4c22_config import day_t_filter_to_df_filter
from t4c22.t4c22_config import day_t_filter_weekdays_daytime_only
from t4c22.t4c22_config import load_inputs
from pathlib import Path
import pendulum

import numpy as np


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets
class T4c22GeometricDataset(torch_geometric.data.Dataset):
    def __init__(
            self,
            root: Path,
            city: str,
            edge_attributes=None,
            split: str = "train",
            fill: int = 1,
            normalize: str = "",
            cachedir: Optional[Path] = None,
            limit: int = None,
            day_t_filter=day_t_filter_weekdays_daytime_only,
            idx=15
    ):
        """Dataset for t4c22 core competition (congestion classes) for one
        city.

        Get 92 items a day (last item of the day then has x loop counter
        data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
        overlapping sampling, but discarding samples going over midnight.

        Missing values in input or labels are represented as nans, use `torch.nan_to_num`.

        Parameters
        ----------,,
        root: basedir for data
        city: "london" / "madrid" / "melbourne"
        edge_attributes: any numerical edge attribute from `road_graph_edges.parquet`
                - parsed_maxspeed
                - speed_kph
                - importance
                - oneway
                - lanes
                - tunnel
                - length_meters
        split: "train" / "test" / ...
        cachedir: location for single item .pt files (created on first access if cachedir is given)
        limit: limit the dataset to at most limit items (for debugging)
        day_t_filter: filter taking day and t as input for filtering the data. Ignored for split=="test".
        """
        super().__init__(root)
        self.root: Path = root

        self.cachedir = cachedir
        self.split = split
        self.fill = fill
        self.normalize = normalize
        self.city = city
        self.limit = limit
        self.day_t_filter = day_t_filter if split != "test" else None

        self.torch_road_graph_mapping = TorchRoadGraphMapping(
            city=city,
            edge_attributes=edge_attributes,
            root=root,
            skip_supersegments=False,
            df_filter=partial(day_t_filter_to_df_filter,
                              filter=day_t_filter) if self.day_t_filter is not None else None,
        )
        # `day_t: List[Tuple[Y-m-d-str,int_0_96]]`
        # TODO most days have even 96 (rolling window over midnight), but probably not necessary because of filtering we do.
        if split == "test":
            num_tests = load_inputs(basedir=self.root, split="test", city=city, day="test", df_filter=None)[
                            "test_idx"].max() + 1
            self.day_t = [("test", t) for t in range(num_tests)]
        else:
            self.day_t = [(day, t) for day in cc_dates(self.root, city=city, split=self.split) for t in range(4, 96) if
                          self.day_t_filter(day, t)]
        import pickle
        self.cluster_map = {}
        with open("data/%s.pkl" % city, 'rb') as f:
            maps = pickle.load(f)
        for i in range(20):
            for day, t, _ in maps[i]:
                self.cluster_map['%s-%d' % (day, t)] = i
        if (len(self.day_t) == 0):
            print("no sample")
            exit(0)

        self.edge_index = self.torch_road_graph_mapping.edge_index
        # self.edge_attr = self.normalize_attr()
        # self.edge_attr = self.torch_road_graph_mapping.edge_attr
        self.edge_attr = self.get_edge_attr()
        self.segment_edge = self.get_segment_edge()
        self.segment_node = self.get_segment_node()

        city_statistics = {"london": [0.0, 7262.0, 348.2629950502402, 289.1281570541699],
                           "madrid": [0.0, 99999.0, 430.7254909879753, 769.3343962046041],
                           "melbourne": [0.0, 11677.0, 172.65683295964914, 217.1803088988486]}

        self.min_volume, self.max_volume, self.mean_volume, self.std = city_statistics[self.city]

    def get_statistics(self):
        pass

    '''
    def get_edge_attr(self):
    
        # ["speed_kph", "parsed_maxspeed", "importance", "length_meters", "counter_distance"]
        edge_attr = torch.zeros((self.torch_road_graph_mapping.edge_attr.shape[0]), 3)
        
        edge_attr[:, 0] = self.torch_road_graph_mapping.edge_attr[:, 3] / self.torch_road_graph_mapping.edge_attr[:, 0]
        edge_attr[:, 0] = edge_attr[:, 0] / 1000 * 3600
        
        return edge_attr
    '''

    def get_segment_edge(self):

        index_list = [[], []]
        for i in range(len(self.torch_road_graph_mapping.supersegments)):
            lists = self.torch_road_graph_mapping.supersegment_to_edges_mapping[i]
            for edge in lists:
                index_list[0].append(i)
                index_list[1].append(self.torch_road_graph_mapping.edge_index_d[edge])

        index = torch.LongTensor(index_list)
        val = torch.ones((index.shape[1]), dtype=torch.float)

        num_row = len(self.torch_road_graph_mapping.supersegments)
        num_col = self.edge_index.shape[1]

        return torch.sparse.FloatTensor(index, val, torch.Size([num_row, num_col]))

    def get_segment_node(self):

        index_list = [[], []]
        for i in range(len(self.torch_road_graph_mapping.supersegments)):
            lists = self.torch_road_graph_mapping.supersegment_to_edges_mapping[i]

            index_list[0].append(i)
            index_list[1].append(self.torch_road_graph_mapping.node_to_int_mapping[lists[0][0]])

            for edge in lists:
                index_list[0].append(i)
                index_list[1].append(self.torch_road_graph_mapping.node_to_int_mapping[edge[1]])

        index = torch.LongTensor(index_list)
        val = torch.ones((index.shape[1]), dtype=torch.float)

        num_row = len(self.torch_road_graph_mapping.supersegments)
        num_col = len(self.torch_road_graph_mapping.nodes)

        return torch.sparse.FloatTensor(index, val, torch.Size([num_row, num_col]))

    def minmax(self, x, min_v, max_v):
        x = x - min_v
        x = x / (max_v - min_v)
        x = x * 2 - 1

        return x

    def zscore(self, x, mean, std):
        x -= mean
        x /= std

        return x

    def get_edge_attr(self):

        # edge_attributes = ["speed_kph", "parsed_maxspeed", "length_meters", "counter_distance", "importance", "highway", "oneway", ]
        num_importance = torch.max(self.torch_road_graph_mapping.edge_attr[:, 4]).item() + 1
        num_highway = torch.max(self.torch_road_graph_mapping.edge_attr[:, 5]).item() + 1
        num_oneway = torch.max(self.torch_road_graph_mapping.edge_attr[:, 6]).item() + 1

        print(num_importance, num_highway, num_oneway)
        self.num_importance = num_importance
        self.num_highway = num_highway
        self.num_oneway = num_oneway

        edge_attr = self.torch_road_graph_mapping.edge_attr[:, :4]
        edge_attr = self.minmax(edge_attr, torch.min(edge_attr, dim=0).values, torch.max(edge_attr, dim=0).values)

        edge_importance = torch.zeros([edge_attr.shape[0], int(num_importance)], dtype=torch.float)
        edge_highway = torch.zeros([edge_attr.shape[0], int(num_highway)], dtype=torch.float)
        edge_oneway = torch.zeros([edge_attr.shape[0], int(num_oneway)], dtype=torch.float)

        edge_importance[
            [i for i in range(edge_attr.shape[0])], [int(j) for j in self.torch_road_graph_mapping.edge_attr[:, 4]]] = 1
        edge_highway[
            [i for i in range(edge_attr.shape[0])], [int(j) for j in self.torch_road_graph_mapping.edge_attr[:, 5]]] = 1
        edge_oneway[
            [i for i in range(edge_attr.shape[0])], [int(j) for j in self.torch_road_graph_mapping.edge_attr[:, 6]]] = 1

        return torch.cat([edge_attr, edge_importance, edge_highway, edge_oneway], dim=1)

    def len(self) -> int:
        if self.limit is not None:
            return min(self.limit, len(self.day_t))
        return len(self.day_t)

    def get(self, idx: int) -> torch_geometric.data.Data:
        """If a cachedir is set, then write data_{day}_{t}.pt on first access
        if it does not yet exist.

        Get 92 items a day (last item of the day then has x loop counter 
        data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
        overlapping sampling, but discarding samples going over midnight
        """

        day, t = self.day_t[idx]

        city = self.city
        basedir = self.root
        split = self.split

        if self.cachedir is not None and (self.cachedir / f"data_{city}_{day}_{t}.pt").exists():
            cache_file = self.cachedir / f"data_{city}_{day}_{t}.pt"
            data = torch.load(cache_file)
            if (self.split == 'test'):
                info_df = pd.read_csv(f"data/{city}_test_info.csv")
                cur_info = info_df[info_df["test_idx"] == data.t]
                t = list(cur_info['t'])[0]
                day = list(cur_info['day'])[0]
                data.t = t
        else:
            # x: 4 time steps of loop counters on nodes
            x = self.torch_road_graph_mapping.load_inputs_day_t(basedir=basedir, city=city, split=split, day=day, t=t,
                                                                idx=idx)
            volume_class, median_speed, max_speed = self.torch_road_graph_mapping.load_speeds_day_t(basedir=basedir,
                                                                                                    city=city,
                                                                                                    split=split,
                                                                                                    day=day, t=t,
                                                                                                    idx=idx)

            # y: congestion classes on edges at +60'
            if self.split == "train":
                y = self.torch_road_graph_mapping.load_cc_labels_day_t(basedir=basedir, city=city, split=split, day=day,
                                                                       t=t, idx=idx)
                eta = self.torch_road_graph_mapping.load_eta_labels_day_t(basedir=basedir, city=city, split=split,
                                                                          day=day, t=t, idx=idx)

            elif self.split == "test":
                y = None
                eta = None

            data = torch_geometric.data.Data(x=x, y=y, eta=eta, volume_class=volume_class, median_speed=median_speed,
                                             max_speed=max_speed, t=t)

            if self.cachedir is not None:
                self.cachedir.mkdir(exist_ok=True, parents=True)
                cache_file = self.cachedir / f"data_{city}_{day}_{t}.pt"
                torch.save(data, cache_file)

        data.cluster = self.cluster_map['%s-%d' % (day, t)]
        data.week = pendulum.parse(day).day_of_week

        # normalize data
        if self.normalize == "mm":
            data.x = self.minmax(data.x, self.min_volume, self.max_volume)
        elif self.normalize == "zs":
            data.x = self.zscore(data.x, self.mean_volume, self.std)
        elif self.normalize == "zsmm":
            data.x = self.zscore(data.x, self.mean_volume, self.std)
            min_v = self.zscore(self.min_volume, self.mean_volume, self.std)
            max_v = self.zscore(self.max_volume, self.mean_volume, self.std)
            data.x = self.minmax(data.x, min_v, max_v)

        # fill data with num
        if self.fill == 1:
            data.x = data.x.nan_to_num(-1)
        elif self.fill == 0:
            data.x = data.x.nan_to_num(0)
        elif self.fill == -1:
            data.x = data.x.nan_to_num(-1)

        return data
