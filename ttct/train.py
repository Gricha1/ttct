import numpy as np
import torch
from loguru import logger
from torch import optim
from utils import gen_mask,KLLoss,split_dataset
from torch.optim import lr_scheduler
from TTCT import TTCT
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

# Comet ML для логирования экспериментов
try:
    from comet_ml import Experiment
    COMET_ML_AVAILABLE = True
except ImportError:
    COMET_ML_AVAILABLE = False
    print("⚠️  Comet ML не установлен. Установите: pip install comet_ml")

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--act_dim', type=int, default=1, help='Action dimension')
parser.add_argument('--context_length', type=int, default=77, help='Context length')
parser.add_argument('--obs_dim', type=int, default=147, help='Observation dimension')
parser.add_argument('--obs_emb_dim', type=int, default=64, help='Observation embedding dimension')
parser.add_argument('--vocab_size', type=int, default=49408, help='Vocabulary size')
parser.add_argument('--trajectory_length', type=int, default=200, help='Trajectory length')
parser.add_argument('--transformer_width', type=int, default=512, help='Transformer width')
parser.add_argument('--transformer_heads', type=int, default=8, help='Transformer heads')
parser.add_argument('--transformer_layers', type=int, default=12, help='Transformer layers')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=194, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--dataset', type=str, default="./dataset/data.pkl")
parser.add_argument('--use_comet', action='store_true', help='Использовать Comet ML для логирования')
parser.add_argument('--comet_project_name', type=str, default='TTCT-Training', help='Имя проекта в Comet ML')
parser.add_argument('--comet_workspace', type=str, default=None, help='Workspace в Comet ML (опционально)')
parser.add_argument('--comet_experiment_name', type=str, default=None, help='Имя эксперимента в Comet ML (опционально)')

args = parser.parse_args()

if __name__ == '__main__':
    # Create a SummaryWriter for logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'./result/{current_time}/log/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Инициализация Comet ML
    comet_experiment = None
    if args.use_comet and COMET_ML_AVAILABLE:
        try:
            comet_experiment = Experiment(
                project_name=args.comet_project_name,
                workspace=args.comet_workspace,
                auto_param_logging=False,  # Будем логировать параметры вручную
                auto_metric_logging=False,  # Будем логировать метрики вручную
                log_code=False,
            )
            print("✅ Comet ML инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации Comet ML: {e}")
            print("   Продолжаем без Comet ML")
            comet_experiment = None
    elif args.use_comet and not COMET_ML_AVAILABLE:
        print("⚠️  Comet ML запрошен, но не установлен. Продолжаем без Comet ML")
    
    # Логирование гиперпараметров в Comet ML
    if comet_experiment:
        hyperparams = {
            'embed_dim': args.embed_dim,
            'act_dim': args.act_dim,
            'context_length': args.context_length,
            'obs_dim': args.obs_dim,
            'obs_emb_dim': args.obs_emb_dim,
            'vocab_size': args.vocab_size,
            'trajectory_length': args.trajectory_length,
            'transformer_width': args.transformer_width,
            'transformer_heads': args.transformer_heads,
            'transformer_layers': args.transformer_layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'dataset': args.dataset,
            'device': str(device),
        }
        
        # Добавляем информацию о GPU если доступно
        if torch.cuda.is_available():
            hyperparams['gpu_name'] = torch.cuda.get_device_name(0)
            hyperparams['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        comet_experiment.log_parameters(hyperparams)
        if args.comet_experiment_name:
            comet_experiment.set_name(args.comet_experiment_name)
        else:
            comet_experiment.set_name(f"TTCT-{current_time}")
        print(f"📊 Comet ML: эксперимент '{comet_experiment.get_name()}' создан")
    
    # Очистка кэша GPU перед началом обучения
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Свободно: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Предупреждение о большом batch_size
        if args.batch_size > 32:
            print(f"⚠️  ВНИМАНИЕ: batch_size={args.batch_size} может быть слишком большим для вашей GPU!")
            print(f"   Рекомендуется использовать --batch_size 16 или меньше")
    trajectory_length = args.trajectory_length
    context_length = args.context_length
    model=TTCT(
        embed_dim = args.embed_dim,
        trajectory_length = args.trajectory_length,
        context_length = args.context_length,
        vocab_size = args.vocab_size,
        transformer_width = args.transformer_width,
        transformer_heads = args.transformer_heads,
        transformer_layers = args.transformer_layers,
        act_dim = args.act_dim,
        obs_dim = args.obs_dim,
        obs_emb_dim = args.obs_emb_dim,
        BERT_PATH='bert-base-uncased',
        device = device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9,0.98),eps=1e-8,weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

    loss_trajectory=KLLoss()
    loss_text=KLLoss()
    total_step=0
    curr_total_loss=0
    curr_auc=0
    curr_TTA_loss=0
    curr_CA_loss=0
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    trainset,testset=split_dataset(args.dataset)
    # Уменьшаем num_workers для экономии памяти
    num_workers = min(4, os.cpu_count() or 1)  # Ограничиваем количество воркеров
    dataloader_train=torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,collate_fn=lambda x:x)
    dataloader_test=torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,collate_fn=lambda x:x)
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader_train, 0):
            model.train()
            transposed_data = list(zip(*data)) 
            obss = transposed_data[0]
            lengths = np.array(transposed_data[3])
            padded_obss = []
            for obs in obss:
                padded_obs = np.pad(obs, ((0, trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
                padded_obss.append(padded_obs)
            padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
            acts = transposed_data[1]
            padded_acts=[]
            padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in acts]
            acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
            TLss = list(transposed_data[2])
            unique_TLs, mask,count=gen_mask(TLss)
            observations = padded_obss.to(device, non_blocking=True)
            NLss=list(transposed_data[4])
            mask=torch.tensor(mask, device=device, dtype=torch.float)
            input_ids = []
            attention_masks = []
            for sent in NLss:
                encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
            attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
            optimizer.zero_grad()
            logits_per_trajectory,CA_loss=model(observations,acts,input_ids,attention_masks,lengths)
            TTA_loss=(loss_trajectory(logits_per_trajectory,mask)+loss_text(logits_per_trajectory.t(),mask.t()))/2
            loss = TTA_loss + CA_loss
            loss.backward()
            optimizer.step()
            curr_total_loss+=loss.item()
            curr_TTA_loss+=TTA_loss.item()
            curr_CA_loss+=CA_loss.item()
            if total_step % 10 == 0:
                mask_cpu=mask.cpu()
                logits_per_trajectory_cpu=logits_per_trajectory.cpu().detach()
                # roc_auc_score требует оба класса (0 и 1) в y_true
                try:
                    y_true = mask_cpu.flatten().numpy()
                    if np.unique(y_true).size < 2:
                        roc_auc = float("nan")
                    else:
                        roc_auc = roc_auc_score(y_true, logits_per_trajectory_cpu.flatten().numpy())
                except Exception:
                    roc_auc = float("nan")
                avg_loss = curr_total_loss / 10
                avg_TTA_loss = curr_TTA_loss/10
                avg_CA_loss = curr_CA_loss/10
                logger.info(f'Epoch: {epoch}, Loss: {avg_loss}, TTA_loss:{avg_TTA_loss}, CA_loss:{avg_CA_loss}, AUC: {roc_auc}')
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_step)
                writer.add_scalar('Loss/Train_total', avg_loss, total_step)
                writer.add_scalar('Loss/Train_CA', avg_CA_loss, total_step)
                writer.add_scalar('Loss/Train_TTA', avg_TTA_loss, total_step)
                writer.add_scalar('AUC', roc_auc, total_step)
                
                # Логирование в Comet ML
                if comet_experiment:
                    comet_experiment.log_metric('train/learning_rate', optimizer.param_groups[0]['lr'], step=total_step)
                    comet_experiment.log_metric('train/loss_total', avg_loss, step=total_step)
                    comet_experiment.log_metric('train/loss_CA', avg_CA_loss, step=total_step)
                    comet_experiment.log_metric('train/loss_TTA', avg_TTA_loss, step=total_step)
                    comet_experiment.log_metric('train/auc', roc_auc, step=total_step)
                    comet_experiment.log_metric('train/epoch', epoch, step=total_step)
                    
                    # Логирование использования GPU памяти (если доступно)
                    if torch.cuda.is_available():
                        comet_experiment.log_metric('system/gpu_memory_allocated', 
                                                   torch.cuda.memory_allocated(0) / 1024**3, step=total_step)
                        comet_experiment.log_metric('system/gpu_memory_reserved', 
                                                   torch.cuda.memory_reserved(0) / 1024**3, step=total_step)
                
                curr_total_loss, curr_auc,curr_TTA_loss,curr_CA_loss= 0, 0, 0, 0
            total_step+=1
            
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_roc_auc=0.0
            test_step=0
            test_TTA_loss=0
            test_CA_loss=0
            for i, data in enumerate(dataloader_test, 0):
                transposed_data = list(zip(*data)) 
                obss = transposed_data[0]
                lengths = np.array(transposed_data[3])
                padded_obss = []
                for obs in obss:
                    padded_obs = np.pad(obs, ((0, trajectory_length - len(obs)), (0, 0), (0, 0), (0, 0)), constant_values=0)
                    padded_obss.append(padded_obs)
                padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
                acts = transposed_data[1]
                padded_acts=[]
                padded_acts = [np.pad(np.array(act, dtype=np.float32), (0, trajectory_length - len(act)), 'constant', constant_values=(-1)) for act in acts]
                acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
                TLss = list(transposed_data[2])
                unique_TLs, mask,count=gen_mask(TLss)
                observations = padded_obss.to(device, non_blocking=True)
                NLss=list(transposed_data[4])
                mask=torch.tensor(mask, device=device, dtype=torch.float)
                input_ids = []
                attention_masks = []
                for sent in NLss:
                    encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                    input_ids.append(encoded_dict['input_ids'])
                    attention_masks.append(encoded_dict['attention_mask'])
                input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
                attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
                logits_per_trajectory,CA_loss=model(observations,acts,input_ids,attention_masks,lengths)
                TTA_loss=(loss_trajectory(logits_per_trajectory,mask)+loss_text(logits_per_trajectory.t(),mask.t()))/2
                mask_cpu=mask.cpu()
                logits_per_trajectory_cpu=logits_per_trajectory.cpu().detach()
                try:
                    y_true = mask_cpu.flatten().numpy()
                    if np.unique(y_true).size >= 2:
                        test_roc_auc += roc_auc_score(y_true, logits_per_trajectory_cpu.flatten().numpy())
                    else:
                        test_roc_auc += 0.0  # один класс — AUC не определён
                except Exception:
                    test_roc_auc += 0.0
                test_step+=1
                test_loss += loss.item()
                test_CA_loss+=CA_loss.item()
                test_TTA_loss+=TTA_loss.item() 
            writer.add_scalar('Test_learning_rate',optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Loss/Test_all', test_loss/test_step, epoch)
            writer.add_scalar('Loss/Test_CA', test_CA_loss/test_step, epoch)
            writer.add_scalar('Loss/Test_TTA', test_TTA_loss/test_step, epoch)
            writer.add_scalar('Test_AUC', test_roc_auc/test_step, epoch)
            
            # Логирование тестовых метрик в Comet ML
            if comet_experiment:
                comet_experiment.log_metric('test/learning_rate', optimizer.param_groups[0]['lr'], step=epoch)
                comet_experiment.log_metric('test/loss_total', test_loss/test_step, step=epoch)
                comet_experiment.log_metric('test/loss_CA', test_CA_loss/test_step, step=epoch)
                comet_experiment.log_metric('test/loss_TTA', test_TTA_loss/test_step, step=epoch)
                comet_experiment.log_metric('test/auc', test_roc_auc/test_step, step=epoch)
                logger.info(f'Comet ML: Эпоха {epoch} залогирована')
        
        scheduler.step()
        checkpoint_dir = f'./result/{current_time}/model/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Сохраняем только последний чекпоинт (удаляем предыдущий для экономии места)
        checkpoint_path = f'./result/{current_time}/model/checkpoint_latest.pt'
        
        # Удаляем предыдущий чекпоинт, если он существует
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить предыдущий чекпоинт: {e}")
        
        # Сохраняем текущий чекпоинт (перезаписываем предыдущий)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Чекпоинт сохранен: {checkpoint_path}')
    
    writer.close()
    
    # Финальный чекпоинт уже сохранен как checkpoint_latest.pt
    checkpoint_path = f'./result/{current_time}/model/checkpoint_latest.pt'
    
    # Логируем финальный чекпоинт в Comet ML (только один раз в конце)
    if comet_experiment:
        if os.path.exists(checkpoint_path):
            comet_experiment.log_asset(checkpoint_path, file_name='checkpoint_latest.pt')
            logger.info(f'Comet ML: Финальный чекпоинт залогирован')
        comet_experiment.end()
        print(f"✅ Comet ML: эксперимент завершен. URL: {comet_experiment.get_url()}")
    
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Чекпоинт: {checkpoint_path}")