import torch
import time
import os
import random
import sys
import yaml
import smart_open
import pickle

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from DL_Lecture.models.sentence_cnn import SentenceCnn
from DL_Lecture.utils.text_prepro import load_snips_data, buildVocab, text_to_indices, sequence_to_tensor


def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = './DL_Lecture/config/text_cnn_snips.yaml'


    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    nb_classes = 0
    # 데이터 로드
    if params['task'] == "SNIPS":
        label_dictionary = {}
        train_seq_path = os.path.join(data_params['train_file'], "seq.in")
        train_label_path = os.path.join(data_params['train_file'], "label")

        val_seq_path = os.path.join(data_params['dev_file'], "seq.in")
        val_label_path = os.path.join(data_params['dev_file'], "label")

        train_x_text, train_y, label_dictionary = load_snips_data(
            train_seq_path, train_label_path, label_dictionary
        )
        nb_classes = max(train_y) + 1
        print("nb_classes: ", nb_classes)

        val_x_text, val_y, label_dictionary = load_snips_data(
            val_seq_path, val_label_path, label_dictionary
        )

    word_id_dict, _ = buildVocab(train_x_text, params['vocab_size']) 
    vocab_size = len(word_id_dict) + 4  
    print("vocabulary size: ", vocab_size)

    for word in word_id_dict.keys():
        word_id_dict[word] += 4  
    word_id_dict['<pad>'] = 0  
    word_id_dict['<unk>'] = 1  
    word_id_dict['<s>'] = 2  
    word_id_dict['</s>'] = 3  

    if params['task'] == "SNIPS":
        train_x = text_to_indices(train_x_text, word_id_dict, params['use_unk_for_oov'])
        val_x = text_to_indices(val_x_text, word_id_dict, params['use_unk_for_oov'])

    nb_pad = int(max(params['model_params_cnn']['filter_lengths']) / 2 + 0.5)

    # Training DataLoader
    train_x = sequence_to_tensor(train_x, nb_paddings=(nb_pad, nb_pad))
    train_y = torch.tensor(train_y)
    training_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=params['batch_size'], shuffle=True, num_workers=4)

    # Validation DataLoader
    val_x = sequence_to_tensor(val_x, nb_paddings=(nb_pad, nb_pad))
    val_y = torch.tensor(val_y)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=params['batch_size'], shuffle=False)

    # 모델 생성
    model = SentenceCnn(nb_classes=nb_classes,
                        filter_lengths=params['model_params_cnn']['filter_lengths'],
                        filter_counts=params['model_params_cnn']['filter_counts'],
                        dropout_rate=params['dropout_rate']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['optimizer_params'][params['optimizer']]['lr'], weight_decay=params['l2_reg_lambda'])
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, gamma=0.99, step_size=1000) 

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir)

     # training 시작
    start_time = time.time()
    highest_val_acc = 0
    global_steps = 0
    print('========================================')
    print("Start training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_correct_cnt = 0
        train_batch_cnt = 0
        model.train()
        for x, y in training_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            step_lr_scheduler.step(global_steps) 

            train_loss += loss.item()
            train_batch_cnt += 1

            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            train_correct_cnt += int(torch.sum(top_pred == y))  

            batch_total = y.size(0)
            batch_correct = int(torch.sum(top_pred == y))
            batch_acc = batch_correct / batch_total

            writer.add_scalar("Batch/Loss", loss.item(), global_steps)
            writer.add_scalar("Batch/Acc", batch_acc, global_steps)

            writer.add_scalar("LR/Learning_rate", step_lr_scheduler.get_last_lr()[0], global_steps)

            global_steps += 1
            if (global_steps) % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, loss.item()))

        train_acc = train_correct_cnt / len(train_y) * 100
        train_ave_loss = train_loss / train_batch_cnt 
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_last_lr()[0])

        # Validation 
        val_correct_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        model.eval()

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_batch_cnt += 1
                _, top_pred = torch.topk(outputs, k=1, dim=-1)
                top_pred = top_pred.squeeze(dim=1)
                val_correct_cnt += int(torch.sum(top_pred == y))

        val_acc = val_correct_cnt / len(val_y) * 100
        val_ave_loss = val_loss / val_batch_cnt
        print("validation dataset accuracy: %.2f" % val_acc)
        writer.add_scalar("Val/Loss", val_ave_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        if val_acc > highest_val_acc:
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)  
            highest_val_acc = val_acc

    vocab_path = os.path.abspath(os.path.join(checkpoint_dir, "vocab"))
    emb_path = os.path.abspath(os.path.join(checkpoint_dir, "emb"))
    labels_path = os.path.abspath(os.path.join(checkpoint_dir, "labels"))
    with smart_open.smart_open(vocab_path, 'wb') as f:
        pickle.dump(word_id_dict, f)
    
    if params['task'] == "SNIPS":
        with smart_open.smart_open(labels_path, 'wb') as f:
            pickle.dump(label_dictionary, f)

        print(f"Highest validation accuracy: {highest_val_acc:.2f}%")


if __name__ == '__main__':
    main()