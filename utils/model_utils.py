import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def accuracy_top_k(output, target, k=1):
    """Computes the Top-k accuracy."""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1)  # Get top-k predictions
        correct = pred.eq(target.view(-1, 1).expand_as(pred))  # Check if target is in top-k
        return correct.float().sum(dim=1).mean().item()  # Compute mean accuracy

def train_and_validate(model, 
                       train_loader, 
                       val_loader, 
                       test_loader, 
                       training_weight_path,
                       criterion, 
                       optimizer, 
                       scheduler, 
                       epochs, 
                       device,
                       early_stopping):
    
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    best_val_loss = float('inf')
    best_model_weights = None
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct_top1, train_correct_top5, total_train = 0, 0, 0, 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]", leave=False)

        for X, y in train_progress:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            # Metrics tracking
            train_loss += loss.item() * X.size(0)
            train_correct_top1 += (preds.argmax(dim=1) == y).sum().item()
            train_correct_top5 += accuracy_top_k(preds, y, k=5) * X.size(0)
            total_train += X.size(0)

            # Update tqdm progress bar
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= total_train
        train_acc_top1 = (train_correct_top1 / total_train) * 100
        train_acc_top5 = (train_correct_top5 / total_train) * 100

        # Validation Phase
        model.eval()
        val_loss, val_correct_top1, val_correct_top5, total_val = 0, 0, 0, 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]", leave=False)

        with torch.no_grad():
            for X, y in val_progress:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)

                # Metrics tracking
                val_loss += loss.item() * X.size(0)
                val_correct_top1 += (preds.argmax(dim=1) == y).sum().item()
                val_correct_top5 += accuracy_top_k(preds, y, k=5) * X.size(0)
                total_val += X.size(0)

                val_progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= total_val
        val_acc_top1 = (val_correct_top1 / total_val) * 100
        val_acc_top5 = (val_correct_top5 / total_val) * 100

        # Adjust learning rate
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter == early_stopping:
                print("Early stopping!")

        # Print Epoch Summary
        print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_loss:.4f}, Top-1 Acc: {train_acc_top1:.2f}%, Top-5 Acc: {train_acc_top5:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Top-1 Acc: {val_acc_top1:.2f}%, Top-5 Acc: {val_acc_top5:.2f}%, LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    if best_model_weights is not None:
        torch.save(best_model_weights, training_weight_path)
        print("Model weights saved!")
        model.load_state_dict(best_model_weights)

    # Test Evaluation
    model.eval()
    test_correct_top1, test_correct_top5, total_test = 0, 0, 0
    test_progress = tqdm(test_loader, desc="Evaluating on Test Set")

    with torch.no_grad():
        for X, y in test_progress:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            test_correct_top1 += (preds.argmax(dim=1) == y).sum().item()
            test_correct_top5 += accuracy_top_k(preds, y, k=5) * X.size(0)
            total_test += X.size(0)

    test_acc_top1 = (test_correct_top1 / total_test) * 100
    test_acc_top5 = (test_correct_top5 / total_test) * 100
    print(f"Test Accuracy -> Top-1: {test_acc_top1:.2f}%, Top-5: {test_acc_top5:.2f}%")

    return test_acc_top1

