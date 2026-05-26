import csv
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset,DataLoader


# Load split
def load_split(dataset):
    outputs = []
    with open(f'./splits/{dataset}_splits.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            _,_,train_videos,test_videos = line.split('/')
            train_videos = train_videos.split(',')
            test_videos = test_videos.split(',')
            test_videos[-1] = test_videos[-1].replace('\n','')
            outputs.append((train_videos,test_videos))
    return outputs


def _fit_length(values,target_length):
    values = np.asarray(values)
    if values.shape[-1]==target_length:
        return values
    if values.shape[-1]>target_length:
        return values[..., :target_length]
    pad_width = [(0,0)] * values.ndim
    pad_width[-1] = (0,target_length-values.shape[-1])
    return np.pad(values,pad_width,mode='edge')


def _normalize_tvsum_annotation_scores(scores):
    scores = np.asarray(scores,dtype=np.float32)
    # TVSum raw annotations are 1-5; CSTA H5 gtscore uses the matching 0-1 scale.
    return np.clip((scores-1.0)/4.0,0.0,1.0).astype(np.float32)


def _load_tvsum_user_scores():
    user_scores = []
    with open('./data/ydata-anno.tsv','r') as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            if not row:
                continue
            curr_user_score = np.array([float(num) for num in row[2].split(',')],dtype=np.float32)
            curr_user_score = _normalize_tvsum_annotation_scores(curr_user_score)
            user_scores.append(curr_user_score[::15])

    outputs = {}
    for start in range(0,len(user_scores),20):
        video_id = start//20 + 1
        outputs[f'video_{video_id}'] = np.stack(user_scores[start:start+20],axis=0)
    return outputs


def _sample_summe_user_summary(user_summary,picks,target_length):
    user_summary = np.asarray(user_summary,dtype=np.float32)
    picks = np.asarray(picks,dtype=np.int64)
    safe_picks = np.clip(picks,0,user_summary.shape[1]-1)
    sampled = user_summary[:, safe_picks]
    return _fit_length(sampled,target_length)


def _make_sample_shot_index(change_points,picks,target_length):
    change_points = np.asarray(change_points)
    picks = np.asarray(picks,dtype=np.int64)
    shot_ends = change_points[:,1]
    shot_index = np.searchsorted(shot_ends,picks,side='left')
    shot_index = np.clip(shot_index,0,len(change_points)-1).astype(np.int64)
    return _fit_length(shot_index,target_length).astype(np.int64)


# Create input,ground truth pair
def load_h5(videos,data_path,dataset_name,return_multi_annot=False,return_shot_info=False):
    features = []
    gtscores = []
    dataset_names = []
    multi_annot_scores = []
    shot_indices = []
    tvsum_user_scores = None

    if return_multi_annot and dataset_name=='TVSum':
        tvsum_user_scores = _load_tvsum_user_scores()

    with h5py.File(data_path,'r') as hdf:
        for video in videos:
            feature = hdf[video]['features'][()]
            gtscore = hdf[video]['gtscore'][()]

            features.append(feature)
            gtscores.append(gtscore)
            dataset_names.append(dataset_name)

            if return_multi_annot:
                if dataset_name=='TVSum':
                    annot_score = _fit_length(tvsum_user_scores[video],len(gtscore))
                elif dataset_name=='SumMe':
                    annot_score = _sample_summe_user_summary(
                        hdf[video]['user_summary'][()],
                        hdf[video]['picks'][()],
                        len(gtscore)
                    )
                else:
                    raise ValueError(f'Unsupported dataset for multi-annot loss: {dataset_name}')
                multi_annot_scores.append(annot_score.astype(np.float32))

            if return_shot_info:
                shot_index = _make_sample_shot_index(
                    hdf[video]['change_points'][()],
                    hdf[video]['picks'][()],
                    len(gtscore)
                )
                shot_indices.append(shot_index)

    outputs = [features,gtscores,dataset_names]
    if return_multi_annot:
        outputs.append(multi_annot_scores)
    if return_shot_info:
        outputs.append(shot_indices)
    return tuple(outputs)


# Create Dataset
class VSdataset(Dataset):
    def __init__(self,data,video_nums,transform=None,return_multi_annot=False,return_shot_info=False):
        features,gtscores,dataset_names = data[:3]
        self.features = features
        self.gtscores = gtscores
        self.dataset_names = dataset_names
        self.video_nums = video_nums
        self.transform = transform
        self.return_multi_annot = return_multi_annot
        self.return_shot_info = return_shot_info

        cursor = 3
        self.multi_annot_scores = None
        self.shot_indices = None
        if self.return_multi_annot:
            self.multi_annot_scores = data[cursor]
            cursor += 1
        if self.return_shot_info:
            self.shot_indices = data[cursor]

    def __len__(self):
        return len(self.video_nums)

    def __getitem__(self,idx):
        output_feature = torch.from_numpy(self.features[idx]).float()
        output_feature = output_feature.unsqueeze(0).expand(3,-1,-1)
        if self.transform is not None:
            output_feature=  self.transform(output_feature)
        sample = [
            torch.unsqueeze(output_feature,0),
            torch.from_numpy(self.gtscores[idx]).float(),
            self.dataset_names[idx],
            self.video_nums[idx]
        ]
        if self.return_multi_annot:
            sample.append(torch.from_numpy(self.multi_annot_scores[idx]).float())
        if self.return_shot_info:
            sample.append(torch.from_numpy(self.shot_indices[idx]).long())
        return tuple(sample)


def collate_fn(sample):
    return sample[0]


# Create Dataloader
def create_dataloader(dataset,config=None):
    loaders = []
    return_multi_annot = bool(getattr(config,'use_multi_annot_loss',False)) if config is not None else False
    return_shot_info = bool(getattr(config,'use_shot_loss',False)) if config is not None else False

    splits = load_split(dataset=dataset)
    data_path = f'./data/eccv16_dataset_{dataset.lower()}_google_pool5.h5'

    for train_videos,test_videos in splits:
        train_data = load_h5(
            videos=train_videos,
            data_path=data_path,
            dataset_name=dataset,
            return_multi_annot=return_multi_annot,
            return_shot_info=return_shot_info
        )
        test_data = load_h5(
            videos=test_videos,
            data_path=data_path,
            dataset_name=dataset,
            return_multi_annot=return_multi_annot,
            return_shot_info=return_shot_info
        )

        train_dataset = VSdataset(
            data=train_data,
            video_nums=train_videos,
            return_multi_annot=return_multi_annot,
            return_shot_info=return_shot_info
        )
        test_dataset = VSdataset(
            data=test_data,
            video_nums=test_videos,
            return_multi_annot=return_multi_annot,
            return_shot_info=return_shot_info
        )
        train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn)
        loaders.append((train_loader,test_loader))
    return loaders
