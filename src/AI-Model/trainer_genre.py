import torch
import time

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss =0.0
    correct =0
    total =0
    start_time =time.time()

    for i, batch in enumerate(train_loader):
        images, additional_features, labels = batch
        images =images.to(device)
        additional_features =additional_features.to(device)
        labels =labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, additional_features)

        if labels.dim() > 1:
            labels =torch.argmax(labels, dim=1)

        perdida= criterion(outputs, labels)
        perdida.backward()
        optimizer.step()
        running_loss +=perdida.item()

        # Predicciones
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = 100*correct/total

        # Cálculo del tiempo restante
        elapsed_time = time.time() - start_time
        batches_remaining = len(train_loader) - i - 1
        avg_batch_time = elapsed_time / (i + 1)
        estimated_time_remaining = avg_batch_time * batches_remaining

        print(f"Batch {i + 1}/{len(train_loader)} - "
              f"Loss: {perdida.item():.4f} - Accuracy: {accuracy:.2f}% - "
              f"Time remaining: {estimated_time_remaining:.2f}s")

    return running_loss / len(train_loader), accuracy

def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss =0.0
    val_preds =[]
    val_probs =[] 
    val_labels =[]
    correct =0
    total =0

    with torch.no_grad():
        for batch in test_loader:
            images, additional_features, labels = batch
            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)
            
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            
            outputs = model(images, additional_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_probs.extend(probs.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy =100 *correct/total if total > 0 else 0

    return val_loss/len(test_loader), val_accuracy, val_preds, val_labels, val_probs


