import shutil

from graph import Grapher
from torch_geometric.utils.convert import from_networkx
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from invoiceGCN import InvoiceGCN


test_output_fd = "/home/nguyen_phuc_toan/Desktop/GraphConvolutionNetwork/dataset/SROIE_2019-20211208T073016Z-001/SROIE_2019/outputs"
shutil.rmtree(test_output_fd)
if not os.path.exists(test_output_fd):
    os.mkdir(test_output_fd)


def load_train_test_split(save_fd):
    train_data = torch.load(os.path.join(save_fd, 'train_data.dataset'))
    test_data = torch.load(os.path.join(save_fd, 'test_data.dataset'))
    return train_data, test_data


def make_info(img_id='584'):
    connect = Grapher(img_id, data_fd='/home/nguyen_phuc_toan/Desktop/GraphConvolutionNetwork/dataset/SROIE_2019-20211208T073016Z-001/SROIE_2019')
    G, _, _ = connect.graph_formation()
    df = connect.relative_distance()
    individual_data = from_networkx(G)
    img_fd = '/home/nguyen_phuc_toan/Desktop/GraphConvolutionNetwork/dataset/SROIE_2019-20211208T073016Z-001/SROIE_2019/raw/img'
    img = cv2.imread(os.path.join(img_fd, "{}.jpg".format(img_id)))[:, :, ::-1]

    return G, df, individual_data, img


train_data, test_data = load_train_test_split(save_fd="/home/nguyen_phuc_toan/Desktop/GraphConvolutionNetwork/dataset/SROIE_2019-20211208T073016Z-001/SROIE_2019/processed")

model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=True)
model.load_state_dict(torch.load('./models/model.pt'))

y_preds = model(test_data).max(dim=1)[1].cpu().numpy()
LABELS = ["company", "address", "date", "total", "other"]
test_batch = test_data.batch.cpu().numpy()
indexes = range(len(test_data.img_id))
print(indexes)
# print(indexes)
for index in indexes:
    start = time.time()
    img_id = test_data.img_id[index]  # not ordering by img_id
    sample_indexes = np.where(test_batch == index)[0]
    y_pred = y_preds[sample_indexes]

    # print("Img index: {}".format(index))
    # print("Img id: {}".format(img_id))
    # print('y_pred: {}'.format(y_pred))
    # print("y_pred: {}".format(Counter(y_pred)))
    G, df, individual_data, img = make_info(img_id)
    try:
        assert len(y_pred) == df.shape[0]

        img2 = np.copy(img)
        for row_index, row in df.iterrows():
            x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
            true_label = row["labels"]

            if isinstance(true_label, str) and true_label != "invoice":
                cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            _y_pred = y_pred[row_index]
            if _y_pred != 4:
                cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
                _label = LABELS[_y_pred]
                cv2.putText(
                    img2, "{}".format(_label), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

        end = time.time()
        # print("\tImage {}: {}".format(img_id, end - start))
        # plt.imshow(img2)
        plt.savefig(os.path.join(test_output_fd, '{}_result.png'.format(img_id)), bbox_inches='tight')
        # plt.savefig('{}_result.png'.format(img_id), bbox_inches='tight')
    except:
        continue
