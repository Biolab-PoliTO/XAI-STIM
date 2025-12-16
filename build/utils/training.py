import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import time
import numpy as np

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, output_path,
                num_epochs=100, checkpoint_interval=10, early_stop_th=10, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.exists(checkpoint_path) and checkpoint_interval > 0:
        os.makedirs(checkpoint_path)

    # Save model architecture to txt file
    arch_path = os.path.join(output_path, "model_architecture.txt")
    with open(arch_path, "w") as f:
        f.write(str(model))
    print(f"Model architecture saved to {arch_path}")

    # Initialize variables
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    start_epoch = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'val_avg_time': []
    }

    # Try to load existing training history
    history_path = os.path.join(output_path, "training_history.csv")
    if os.path.exists(history_path):
        print(f"Found existing training history at {history_path}")
        try:
            history_df = pd.read_csv(history_path)
            history['epoch'] = history_df['epoch'].tolist()
            history['train_loss'] = history_df['train_loss'].tolist()
            history['train_acc'] = history_df['train_acc'].tolist()
            history['val_loss'] = history_df['val_loss'].tolist()
            history['val_acc'] = history_df['val_acc'].tolist()
            history['test_loss'] = history_df['val_loss'].tolist()
            history['test_acc'] = history_df['val_acc'].tolist()
            history['val_avg_time'] = history_df['val_avg_time'].tolist()
            
            start_epoch = len(history['epoch'])
            best_val_acc = max(history['val_acc'])
            best_val_loss = min(history['val_loss'])
            print(f"Resuming from epoch {start_epoch}, best_val_acc: {best_val_acc:.4f}, best_val_loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading training history: {e}")
            print("Starting training from scratch")

    # Try to load existing model checkpoint
    best_model_path = os.path.join(output_path, "best_model.pt")
    latest_checkpoint = None
    
    # Look for the latest checkpoint
    if checkpoint_interval > 0 and os.path.exists(checkpoint_path):
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint_it_") and f.endswith(".pt")]
        if checkpoint_files:
            # Extract epoch numbers and find the latest
            checkpoint_epochs = []
            for cf in checkpoint_files:
                try:
                    epoch_num = int(cf.replace("checkpoint_it_", "").replace(".pt", ""))
                    checkpoint_epochs.append((epoch_num, cf))
                except:
                    continue
            if checkpoint_epochs:
                latest_epoch, latest_file = max(checkpoint_epochs, key=lambda x: x[0])
                latest_checkpoint = os.path.join(checkpoint_path, latest_file)
                print(f"Found latest checkpoint: {latest_checkpoint} (epoch {latest_epoch})")

    # Load model weights (prefer best model if available, otherwise latest checkpoint)
    model_to_load = None
    if os.path.exists(best_model_path):
        model_to_load = best_model_path
        print(f"Loading best model from {best_model_path}")
    elif latest_checkpoint:
        model_to_load = latest_checkpoint
        print(f"Loading latest checkpoint from {latest_checkpoint}")
    
    if model_to_load:
        try:
            model.load_state_dict(torch.load(model_to_load, map_location=device))
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Starting training with initial model weights")

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)

    scaler = torch.amp.GradScaler("cuda") ###

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs, _ = model(inputs)

            # Prepare targets and (optionally) model outputs depending on loss type
            # Support: BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, NLLLoss
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                num_classes = outputs.shape[1]
                targets = F.one_hot(labels, num_classes=num_classes).float()
                outputs_for_loss = outputs
            elif isinstance(criterion, nn.BCELoss):
                num_classes = outputs.shape[1]
                targets = F.one_hot(labels, num_classes=num_classes).float()
                # BCELoss expects probabilities in [0,1]
                outputs_for_loss = torch.sigmoid(outputs)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                # CrossEntropyLoss expects class indices (N,) long
                targets = labels.long().squeeze()
                outputs_for_loss = outputs
            elif isinstance(criterion, nn.NLLLoss):
                targets = labels.long().squeeze()
                outputs_for_loss = F.log_softmax(outputs, dim=1)
            else:
                # Fallback: try BCEWithLogits-like behavior
                try:
                    num_classes = outputs.shape[1]
                    targets = F.one_hot(labels, num_classes=num_classes).float()
                    outputs_for_loss = outputs
                except Exception:
                    targets = labels
                    outputs_for_loss = outputs

            loss = criterion(outputs_for_loss, targets)

            import math
            if math.isnan(loss):
                print('Error')
                
            scaler.scale(loss).backward() ###
            scaler.unscale_(optimizer) ###
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ###
            scaler.step(optimizer) ###
            scaler.update() ###

            running_loss += loss.item()

            # Compute predicted labels and true labels for accuracy
            _, predicted = torch.max(outputs.data, 1)
            # true labels as indices
            if targets.dim() == 2:
                true_idx = targets.argmax(dim=1)
            else:
                true_idx = targets.long()
            total += true_idx.size(0)
            correct += (predicted == true_idx).sum().item()

        if checkpoint_interval > 0 and (epoch+1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"checkpoint_it_{epoch+1}.pt"))
            print(f"Iteration {epoch+1}: checkpoint saved to {checkpoint_path}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_acc, val_loss, val_avg_time = evaluate_model(model, val_loader, criterion, device)
        test_acc, test_loss, _ = evaluate_model(model, test_loader, criterion, device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #scheduler.step(val_loss)
            scheduler.step(train_loss)
        else:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Logging history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['val_avg_time'].append(val_avg_time)

        # Save best model based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

        # Early stopping based on val loss improvement
        if best_val_loss - val_loss > 0.001:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No significant improvement in val loss. Early stop count: {early_stop_counter}/{early_stop_th}")

        if early_stop_counter >= early_stop_th:
            print("Early stopping triggered.")
            break
        
        # Save training history to CSV
        history_df = pd.DataFrame(history)
        history_path = os.path.join(output_path, "training_history.csv")
        history_df.to_csv(history_path, index=False)

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_path, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    return model, history, best_val_acc, best_val_loss


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)

            # Prepare targets and (optionally) model outputs depending on loss type
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                num_classes = outputs.shape[1]
                targets = F.one_hot(labels, num_classes=num_classes).float()
                outputs_for_loss = outputs
            elif isinstance(criterion, nn.BCELoss):
                num_classes = outputs.shape[1]
                targets = F.one_hot(labels, num_classes=num_classes).float()
                outputs_for_loss = torch.sigmoid(outputs)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                targets = labels.long().squeeze()
                outputs_for_loss = outputs
            elif isinstance(criterion, nn.NLLLoss):
                targets = labels.long().squeeze()
                outputs_for_loss = F.log_softmax(outputs, dim=1)
            else:
                try:
                    num_classes = outputs.shape[1]
                    targets = F.one_hot(labels, num_classes=num_classes).float()
                    outputs_for_loss = outputs
                except Exception:
                    targets = labels
                    outputs_for_loss = outputs

            loss = criterion(outputs_for_loss, targets)

            import math
            if math.isnan(loss):
                print('Error')

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            if targets.dim() == 2:
                true_idx = targets.argmax(dim=1)
            else:
                true_idx = targets.long()
            total += true_idx.size(0)
            correct += (predicted == true_idx).sum().item()
    end_time = time.time()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    avg_time = (end_time - start_time) / len(data_loader)
    return accuracy, avg_loss, avg_time

def main_training(X_train, y_train, X_val, y_val, model, lr, bs, scheduler, optimizer, criterion, output_dir, X_test=None, y_test=None):

    rem_batches_train = X_train.shape[0] % bs
    if rem_batches_train == 1:
        X_train = X_train[:-1]
        y_train = y_train[:-1]
    rem_batches_val = X_val.shape[0] % bs
    if rem_batches_val == 1:
        X_val = X_val[:-1]
        y_val = y_val[:-1]
    if X_test is not None and y_test is not None:
        rem_batches_test = X_test.shape[0] % bs
        if rem_batches_test == 1:
            X_test = X_test[:-1]
            y_test = y_test[:-1]

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss() ###

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    del X_train
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    del X_val
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    class_sample_count = np.bincount(y_train)
    weights = 1. / class_sample_count
    sample_weights = weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True)
    del train_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    del val_dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    
    # Train model
    trained_model, history, best_val_acc, best_val_loss = train_model(model, 
                                                                train_loader, 
                                                                val_loader, 
                                                                test_loader,
                                                                optimizer, 
                                                                criterion, 
                                                                output_path=os.path.join(output_dir,f'training_progress_lr{lr}_bs{bs}'),
                                                                num_epochs=100, 
                                                                checkpoint_interval=5,
                                                                early_stop_th= 20, #50, #20,
                                                                scheduler=scheduler,
                                                                )
    output = {'history': history, 'best_val_acc': best_val_acc, 'best_val_loss': best_val_loss}

    if X_test is not None and y_test is not None:
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        del X_test
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        del test_dataset
        test_accuracy, test_loss, test_time = evaluate_model(trained_model, test_loader, criterion, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print("-" * 80)
        print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}, Test time (per 1-second segment): {test_time}')
        print("=" * 80)

    return trained_model, output