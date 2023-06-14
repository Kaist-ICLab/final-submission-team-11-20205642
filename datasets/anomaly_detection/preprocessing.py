f_name = 'WalkingAceleration5_2700_5920_5979.txt'
ts = pd.read_csv(f'datasets/anomaly_detection/{f_name}', header=None, dtype=float).values
idx = np.array(f_name.split('.')[0].split('_')[-3:]).astype(int)
train_idx, label_start, label_end = idx[0], idx[1], idx[2]+1
labels = np.zeros(ts.shape[0], dtype=int)
labels[label_start:label_end] = 1

train_ts = ts[:train_idx]
test_ts = ts[train_idx:]
labels = labels[train_idx:]

pd.DataFrame(train_ts).to_csv('datasets/anomaly_detection/Walking5/train.csv', index=False)
pd.DataFrame(test_ts).to_csv('datasets/anomaly_detection/Walking5/test.csv', index=False)
pd.DataFrame(labels).to_csv('datasets/anomaly_detection/Walking5/labels.csv', index=False)