from utils import *
from model import *
from config import *
from model_dev import dev

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCH):
        for b, (input, target, mask) in enumerate(loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            y_pred = model(input, mask)

            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if b % 10 == 0:
            #     print('>> epoch:', e, 'loss:', loss.item())
        dev_report = dev(model)

        f1_score = round(dev_report['macro avg']['f1-score'], 4)
        print('>> epoch:', e, 'loss:', loss.item(), 'dev_f1:', f1_score)

        # 保存模型参数
        if f1_score > 0.6:
            torch.save(model.state_dict(),MODEL_DIR + f'model_{e}.pth')
        # torch.save(model, MODEL_DIR + f'model_{e}.pth')
