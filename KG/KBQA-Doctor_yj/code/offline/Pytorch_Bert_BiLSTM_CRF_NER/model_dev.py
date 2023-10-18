from utils import *
from model import *
from config import *

def dev(model):
    dataset = Dataset('dev')
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    with torch.no_grad():

        y_true_list = []
        y_pred_list = []

        id2label, _ = get_label()

        for b, (input, target, mask) in enumerate(loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)
            
            y_pred = model(input, mask)

            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y,m in zip(target, mask):
                y_true_list.append([id2label[i] for i in y[m==True].tolist()])

        return report(y_true_list, y_pred_list)
