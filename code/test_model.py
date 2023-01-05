from config import *
from metrics import AverageMeter

def test_model(test_loader, model):
    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    test_f1_meter = AverageMeter()
    test_EER_meter = AverageMeter()
    all_matrix = [[0, 0], [0, 0]]
    full_vec = []
    full_labels = []
    for i, batch in enumerate(tqdm(test_loader)):
        # Move batch to device if device != 'cpu'
        wav = batch[0].to(CFG.device)
        # length = batch['length'].to(device)
        label = batch[1].to(CFG.device)
        label = label.reshape(len(label))
        label = torch.tensor(label).long()
        wav = wav.squeeze()
        with torch.no_grad():
            output = model(wav)
            out = output.argmax(dim=-1).cpu().numpy()
            out2 = output.softmax(dim=1)
            out2 = out2.transpose(0, 1)
            out2 = out2[1][:]
            out2 = out2.cpu().numpy()
            labels = label.cpu().numpy()
            full_vec.extend(out2)
            full_labels.extend(labels)
        # print(f'output :{output.argmax(dim=-1)}, label : {label}')

        matches = (output.argmax(dim=-1) == label).float().mean()
        f1 = f1_score(output.argmax(dim=-1).cpu(), label.cpu(), average='weighted')
        eer = compute_eer(labels, out)
        # print(f'Test:{matches.item()}')

        test_loss_meter.update(loss.item(), len(batch[0]))
        test_accuracy_meter.update(matches.item(), len(batch[0]))
        # test_EER_meter.update(eer[0],              len(batch[0]))
        test_f1_meter.update(f1, len(batch[0]))

        matrix = confusion_matrix(labels, out, labels=[0, 1])
        all_matrix += matrix
        # print(f'F1 :{f1} , EER: {eer}')
        # print(f'Confusion Matrix of all:{all_matrix}')\

    mDCF = minDCF(all_matrix)
    validation_MinDCF_meter.update(mDCF)

    print(f'Confusion Matrix of all:{all_matrix}')
    print(f'minDCF : {mDCF}% ')
    plt.title('ROC CURVE WITH EER')
    fpr, tpr, _ = metrics.roc_curve(full_labels, full_vec)
    EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    print(f'EER : {EER}%')

    auc = metrics.roc_auc_score(full_labels, full_vec)

    # display.clear_output()
    print(f'Confusion Matrix of all:{all_matrix}')
    print(f'Accuracy on Test Dataset:{test_accuracy_meter.avg} !')