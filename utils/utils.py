import os
from pathlib import Path
import torch

DEVICE = torch.device('cuda')


def get_paths(path, str_format=True):
    """Возвращает список объектов в папке path"""
    image_names = []
    for dir_name, _, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dir_name, filename)
            image_names.append(full_path)
    if str_format:
        return image_names
    else:
        return [Path(i) for i in image_names]


def predict(model, test_loader,type_of_data='train'):
    """
    Возвращает список предсказаний модели для выборки.
    model - модель,
    test-loader - Датасет для разметки.
    """
    with torch.no_grad():
        logits = []
        if type_of_data=='train':
            for inputs, _ in test_loader:
                inputs = inputs.to(DEVICE)
                model.eval()
                outputs = model(inputs).squeeze(dim=1).cpu()
                logits.append(outputs)
        else:
            for inputs in test_loader:
                inputs = inputs.to(DEVICE)
                model.eval()
                outputs = model(inputs).squeeze(dim=1).cpu()
                logits.append(outputs)

    probs = (torch.sigmoid(torch.cat(logits)) > 0.5).int().numpy()
    return probs


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Intersection over Union - метрика.
    outputs - выходы модели, размерность height X weight
    labels - маска изображения.
    """
    labels = labels.int()
    outputs = (outputs > 0.5).int()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # smooth  devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  #


def score_model(model, metric, data):
    """Подсчет метрики на всем датасете"""
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        X_batch = X_batch.to(DEVICE)
        Y_pred = torch.sigmoid(model(X_batch))
        scores += metric(Y_pred, Y_label.to(DEVICE)).mean().item()
    return scores / len(data)


def foreground_extr(img_pil,model):
    """Наложение маски на изображение"""
    torch_img = trf_rgb_imagenet(img_pil).to(DEVICE)
    model_outp = model(torch_img.unsqueeze(dim=0)).squeeze().detach().cpu()
    mask_classes=np.array((model_outp>0.5)).astype(int).astype(np.uint8)
    img_array = np.array(default_transf(img_pil).squeeze()).transpose((1,2,0))
    bit_mask = (mask_classes!=0).astype(np.uint8)
    bit_img = cv2.bitwise_and(img_array,img_array,mask=bit_mask)
    return bit_img


def visualize(image,model):
    """Визуализация масок"""
    masked_image = foreground_extr(image,model)
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(8, 8),
                           sharey=True, sharex=True)
    image = default_transf(image).numpy().transpose((1,2,0))
    ax[0].imshow(image)
    ax[1].imshow(masked_image)