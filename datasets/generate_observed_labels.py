import numpy as np
from collections.abc import Sequence


def class_generate_observed_labels(mode, all_labels, pos_param, neg_param, add_one_label=False):
    n_class = all_labels.shape[1]
    n_sample = all_labels.shape[0]
    obs_labels = np.zeros((n_sample, n_class), dtype=np.float32)
    for idx_c in range(n_class):
        sub_label = np.zeros(n_sample)
        c_label = all_labels[:, idx_c]
        pos_idx = np.where(c_label > 0)[0]
        neg_idx = np.where(c_label < 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        if mode == "fix_class_num":
            assert (1 <= pos_param < n_sample) and (1 <= neg_param < n_sample)
            pos_num = min(int(pos_param), n_pos)
            neg_num = min(int(neg_param), n_neg)
        elif mode == "fix_class_prop":
            assert (0 < pos_param <= 1) and (0 < neg_param <= 1)
            pos_num = max(int(np.ceil(n_pos * pos_param)), 1)
            neg_num = max(int(np.ceil(n_neg * neg_param)), 1)
        if n_pos > 0:
            sub_pos_idx = np.random.choice(pos_idx, pos_num, False)
            sub_label[sub_pos_idx] = c_label[sub_pos_idx]
        if n_neg > 0:
            sub_neg_idx = np.random.choice(neg_idx, neg_num, False)
            sub_label[sub_neg_idx] = c_label[sub_neg_idx]
        obs_labels[:, idx_c] = sub_label
    if add_one_label:
        for i, x in enumerate(obs_labels):
            if not np.any(x == 1):
                pos_idx = np.where(all_labels[i] > 0)[0]
                select_pos_idx = np.random.choice(pos_idx, 1)
                obs_labels[i, select_pos_idx] = 1
            if not np.any(x == -1):
                neg_idx = np.where(all_labels[i] < 0)[0]
                select_neg_idx = np.random.choice(neg_idx, 1)
                obs_labels[i, select_neg_idx] = -1
    return obs_labels


def img_generate_observed_labels(all_labels, num_pos, num_neg):
    assert isinstance(num_pos, int) and isinstance(num_neg, int)
    n_class = all_labels.shape[1]
    n_sample = all_labels.shape[0]
    obs_labels = np.zeros((n_sample, n_class), dtype=np.float32)
    for idx_m in range(n_sample):
        sub_label = np.zeros(n_class)
        m_label = all_labels[idx_m, :]
        pos_idx = np.where(m_label > 0)[0]
        n_pos = len(pos_idx)
        if n_pos > 0:
            if num_pos == -1:
                n_sub_pos = n_pos
            else:
                n_sub_pos = num_pos
            np.random.shuffle(pos_idx)
            sub_pos_idx = pos_idx[:n_sub_pos]
            sub_label[sub_pos_idx] = m_label[sub_pos_idx]
        neg_idx = np.where(m_label < 0)[0]
        n_neg = len(neg_idx)
        if n_neg > 0:
            if num_neg == -1:
                n_sub_neg = n_neg
            else:
                n_sub_neg = num_neg
            np.random.shuffle(neg_idx)
            sub_neg_idx = neg_idx[:n_sub_neg]
            sub_label[sub_neg_idx] = m_label[sub_neg_idx]
        obs_labels[idx_m, :] = sub_label
    return obs_labels


def global_random_generate_observed_labels(all_labels, prop):
    assert (0 < prop <= 1)
    mask = np.random.random(all_labels.shape)
    mask[mask < prop] = 1
    mask[mask < 1] = 0
    obs_labels = mask * all_labels
    return obs_labels


def generate_observed_labels(all_labels, sample_mode='global_random', num_obs=1.0):
    if sample_mode in ['fix_class_prop', 'global_random']:
        num_obs = [float(x) for x in num_obs]
    else:
        num_obs = [int(x) for x in num_obs]
    np.random.seed(0)

    if not isinstance(num_obs, (int, float, Sequence)):
        raise TypeError("num_obs should be int, float or sequence. Got {}".format(type(num_obs)))
    if isinstance(num_obs, (int, float)):
        pos_obs, neg_obs = num_obs, num_obs
    elif len(num_obs) < 2:
        pos_obs, neg_obs = num_obs[0], num_obs[0]
    else:
        pos_obs, neg_obs = num_obs[0], num_obs[1]
    if 'class' in sample_mode:
        obs_labels = class_generate_observed_labels(sample_mode, all_labels, pos_obs, neg_obs)
    elif 'img' in sample_mode:
        obs_labels = img_generate_observed_labels(all_labels, pos_obs, neg_obs)
    elif 'global' in sample_mode:
        obs_labels = global_random_generate_observed_labels(all_labels, pos_obs)

    return obs_labels