import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import clip
from timm import create_model

# Constants
BATCH_SIZE = 64
LEARNING_RATE = 3e-5
NUM_EPOCHS = 30
EMBEDDING_DIM = 512
TEMPERATURE = 0.07
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Advanced transforms for data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Updated ImageRetrievalDataset with more functionality
class ImageRetrievalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False, return_path=False):
        self.data_info = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.return_path = return_path
        
        if not is_test and 'class' in self.data_info.columns:
            self.labels = self.data_info['class'].values
            # Create class-to-idx and idx-to-class mappings
            self.classes = sorted(list(set(self.labels)))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        self.file_names = self.data_info['filename'].values if 'filename' in self.data_info.columns else self.data_info.iloc[:, 1].values
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        
        image = Image.open(img_path).convert('RGB')
        transformed_image = self.transform(image) if self.transform else image
        
        file_id = os.path.splitext(img_name)[0]  # Remove file extension to get ID
        
        if not self.is_test and hasattr(self, 'labels'):
            label = self.labels[idx]
            label_idx = self.class_to_idx[label]
            if self.return_path:
                return transformed_image, label_idx, file_id, img_path
            return transformed_image, label_idx, file_id
        else:
            if self.return_path:
                return transformed_image, file_id, img_path
            return transformed_image, file_id

# === APPROACH 1: METRIC LEARNING WITH TRIPLET LOSS ===

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        positive_dist = torch.norm(anchor - positive, dim=1)
        negative_dist = torch.norm(anchor - negative, dim=1)
        losses = F.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()

class MetricLearningModel(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super(MetricLearningModel, self).__init__()
        # Use EfficientNet as the backbone
        self.backbone = create_model('efficientnet_b0', pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def train_metric_learning_model(train_loader, val_loader, epochs=NUM_EPOCHS):
    model = MetricLearningModel().to(DEVICE)
    triplet_loss = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Create triplets
            embeddings = model(images)
            triplet_indices = []
            
            for i in range(len(labels)):
                anchor_label = labels[i].item()
                # Find positive (same class as anchor)
                positive_indices = [j for j in range(len(labels)) if labels[j].item() == anchor_label and j != i]
                # Find negative (different class than anchor)
                negative_indices = [j for j in range(len(labels)) if labels[j].item() != anchor_label]
                
                if positive_indices and negative_indices:
                    positive_idx = np.random.choice(positive_indices)
                    negative_idx = np.random.choice(negative_indices)
                    triplet_indices.append((i, positive_idx, negative_idx))
            
            if triplet_indices:
                anchors = torch.stack([embeddings[i] for i, _, _ in triplet_indices])
                positives = torch.stack([embeddings[j] for _, j, _ in triplet_indices])
                negatives = torch.stack([embeddings[k] for _, _, k in triplet_indices])
                
                optimizer.zero_grad()
                loss = triplet_loss(anchors, positives, negatives)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                embeddings = model(images)
                triplet_indices = []
                
                for i in range(len(labels)):
                    anchor_label = labels[i].item()
                    positive_indices = [j for j in range(len(labels)) if labels[j].item() == anchor_label and j != i]
                    negative_indices = [j for j in range(len(labels)) if labels[j].item() != anchor_label]
                    
                    if positive_indices and negative_indices:
                        positive_idx = np.random.choice(positive_indices)
                        negative_idx = np.random.choice(negative_indices)
                        triplet_indices.append((i, positive_idx, negative_idx))
                
                if triplet_indices:
                    anchors = torch.stack([embeddings[i] for i, _, _ in triplet_indices])
                    positives = torch.stack([embeddings[j] for _, j, _ in triplet_indices])
                    negatives = torch.stack([embeddings[k] for _, _, k in triplet_indices])
                    
                    loss = triplet_loss(anchors, positives, negatives)
                    val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_metric_model.pth')
    
    return model

# === APPROACH 2: SELF-SUPERVISED LEARNING WITH SIMCLR ===

class SimCLRModel(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super(SimCLRModel, self).__init__()
        # Use ResNet as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return embeddings

class NTXentLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        z = torch.cat([z_i, z_j], dim=0)
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t().contiguous()) / self.temperature
        
        # Mask for removing self-similarity
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # Positive samples: similarity between augmented pairs
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
        
        # Mask out self-comparisons
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool).to(DEVICE)
        mask = mask.fill_diagonal_(0)
        
        # Negative samples: all other similarities
        negative_samples = sim[mask].reshape(2 * batch_size, -1)
        
        # Concatenate positive and negative samples
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        
        # Labels: positives are the 0th element for all examples
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(DEVICE)
        
        return self.criterion(logits, labels)

# SimCLR augmentation strategy
class SimCLRDataTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
        
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)

# Self-supervised Dataset
class SelfSupervisedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, *rest = self.dataset[index]
        if self.transform:
            image1, image2 = self.transform(image)
            return (image1, image2), *rest
        return image, *rest
    
    def __len__(self):
        return len(self.dataset)

def train_simclr_model(train_loader, val_loader, epochs=NUM_EPOCHS):
    model = SimCLRModel().to(DEVICE)
    criterion = NTXentLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, ((images1, images2), _, _) in enumerate(tqdm(train_loader)):
            images1, images2 = images1.to(DEVICE), images2.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Get embeddings
            z_i = model(images1)
            z_j = model(images2)
            
            # Compute loss
            loss = criterion(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, ((images1, images2), _, _) in enumerate(val_loader):
                images1, images2 = images1.to(DEVICE), images2.to(DEVICE)
                
                # Get embeddings
                z_i = model(images1)
                z_j = model(images2)
                
                # Compute loss
                loss = criterion(z_i, z_j)
                
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_simclr_model.pth')
    
    return model

# === APPROACH 3: SEMI-SUPERVISED LEARNING ===

def get_pseudo_labels(unlabeled_loader, model):
    model.eval()
    features_list = []
    paths_list = []
    
    with torch.no_grad():
        for images, _, paths in tqdm(unlabeled_loader):
            images = images.to(DEVICE)
            features = model(images)
            features_list.append(features.cpu().numpy())
            paths_list.extend(paths)
    
    features_array = np.vstack(features_list)
    
    # Use K-means clustering to get pseudo-labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=100, random_state=0).fit(features_array)
    pseudo_labels = kmeans.labels_
    
    return dict(zip(paths_list, pseudo_labels))

class SemiSupervisedModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super(SemiSupervisedModel, self).__init__()
        # Use ViT as the backbone
        self.backbone = create_model('vit_base_patch16_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
        # Classification head
        self.classification_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        
        if return_features:
            return embeddings
        
        logits = self.classification_head(embeddings)
        return logits, embeddings

def train_semi_supervised_model(labeled_loader, unlabeled_loader, val_loader, num_classes, epochs=NUM_EPOCHS):
    model = SemiSupervisedModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Phase 1: Train on labeled data
        for batch_idx, (images, labels, _) in enumerate(tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{epochs} - Labeled")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Get pseudo-labels for unlabeled data
        if epoch % 5 == 0:  # Update pseudo-labels every 5 epochs
            pseudo_labels = get_pseudo_labels(unlabeled_loader, model)
        
        # Phase 2: Train on unlabeled data with pseudo-labels
        for batch_idx, (images, _, paths) in enumerate(tqdm(unlabeled_loader, desc=f"Epoch {epoch+1}/{epochs} - Unlabeled")):
            images = images.to(DEVICE)
            
            # Get pseudo-labels for this batch
            batch_pseudo_labels = torch.tensor([pseudo_labels[path] for path in paths]).to(DEVICE)
            
            optimizer.zero_grad()
            
            logits, _ = model(images)
            loss = criterion(logits, batch_pseudo_labels)
            
            # Use a confidence threshold
            confidence = F.softmax(logits, dim=1).max(1)[0]
            mask = confidence > 0.7  # Only use confident predictions
            if mask.sum() > 0:
                loss = loss * mask.float()
                loss = loss.sum() / mask.sum()
                
                loss.backward()
                optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                logits, _ = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(labeled_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_semi_supervised_model.pth')
    
    return model

# === APPROACH 4: PRETRAINED CLIP MODEL ===

def extract_clip_features(data_loader):
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    features_list = []
    labels_list = []
    ids_list = []
    
    with torch.no_grad():
        for images, labels, file_ids in tqdm(data_loader):
            if isinstance(images, list):  # Handle multiple views
                images = images[0]
            
            images = images.to(DEVICE)
            features = model.encode_image(images)
            
            features_list.append(features.cpu().numpy())
            if not isinstance(labels, list):  # Handle test data
                labels_list.append(labels.numpy())
            ids_list.extend(file_ids)
    
    features_array = np.vstack(features_list)
    if labels_list:
        labels_array = np.concatenate(labels_list)
        return features_array, labels_array, ids_list
    else:
        return features_array, None, ids_list

# === APPROACH 5: ENSEMBLE OF MULTIPLE MODELS ===

def ensemble_predictions(models, test_loader, num_classes):
    predictions = []
    
    for model in models:
        model.eval()
        model_preds = []
        
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(DEVICE)
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get logits if tuple
                
                # Convert to probabilities
                probs = F.softmax(outputs, dim=1)
                model_preds.append(probs.cpu().numpy())
        
        model_preds = np.vstack(model_preds)
        predictions.append(model_preds)
    
    # Average predictions
    ensemble_preds = np.mean(predictions, axis=0)
    return ensemble_preds

# === APPROACH 6: SWIN TRANSFORMER MODEL ===

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM):
        super(SwinTransformerModel, self).__init__()
        # Use Swin Transformer as backbone
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        
        if return_features:
            return embeddings
        
        logits = self.classifier(embeddings)
        return logits, embeddings

# === RETRIEVAL FUNCTIONS ===

def calculate_map5(predictions, true_labels):
    """
    Calculate Mean Average Precision @ 5 (MAP@5)
    
    Args:
        predictions: List of lists where each sublist contains the top 5 predicted class indices for an image
        true_labels: List of true class indices for each image
        
    Returns:
        MAP@5 score
    """
    ap_sum = 0.0
    
    for i, (preds, true_label) in enumerate(zip(predictions, true_labels)):
        precision_sum = 0.0
        found = False
        
        for k, pred in enumerate(preds[:5]):
            if pred == true_label and not found:
                # Compute precision at this position (k+1)
                precision_at_k = 1.0 / (k + 1)
                precision_sum += precision_at_k
                found = True
        
        if found:
            ap_sum += precision_sum
    
    return ap_sum / len(true_labels)

def retrieve_images(query_features, gallery_features, gallery_labels, k=5):
    """
    Retrieve top-k similar images for each query image
    
    Args:
        query_features: Features of query images [num_queries, feat_dim]
        gallery_features: Features of gallery images [num_gallery, feat_dim]
        gallery_labels: Labels of gallery images [num_gallery]
        k: Number of images to retrieve
    
    Returns:
        List of lists where each sublist contains the top-k predicted labels for a query
    """
    # Compute similarity between query and gallery features
    sim = np.dot(query_features, gallery_features.T)
    
    # Get top-k similar images
    top_k_indices = np.argsort(-sim, axis=1)[:, :k]
    
    # Get corresponding labels
    predictions = []
    for indices in top_k_indices:
        pred_labels = [gallery_labels[idx] for idx in indices]
        predictions.append(pred_labels)
    
    return predictions

def save_submission(predictions, test_ids, output_file):
    """
    Save predictions to a CSV file in the required format
    
    Args:
        predictions: List of lists where each sublist contains the top-5 predicted class indices for an image
        test_ids: List of image IDs
        output_file: Output CSV file path
    """
    submission = pd.DataFrame(columns=['image_id', 'class'])
    
    for i, (image_id, preds) in enumerate(zip(test_ids, predictions)):
        # For each image, create up to 5 rows with predictions
        for j, pred in enumerate(preds[:5]):
            if pred == -1:  # Skip -1 predictions
                continue
            submission = submission.append({
                'image_id': image_id,
                'class': pred
            }, ignore_index=True)
    
    submission.to_csv(output_file, index=False)

# === MAIN FUNCTION ===

def main():
    # Set paths
    train_csv = 'train.csv'
    test_csv = 'test.csv'
    train_img_dir = 'train_images'
    test_img_dir = 'test_images'
    unlabeled_csv = 'unlabeled.csv'  # If available
    unlabeled_img_dir = 'unlabeled_images'  # If available
    
    # Create datasets
    train_dataset = ImageRetrievalDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = ImageRetrievalDataset(train_csv, train_img_dir, transform=val_transform)
    test_dataset = ImageRetrievalDataset(test_csv, test_img_dir, transform=val_transform, is_test=True)
    
    # Split train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create unlabeled dataset if available
    try:
        unlabeled_dataset = ImageRetrievalDataset(unlabeled_csv, unlabeled_img_dir, transform=val_transform, is_test=True, return_path=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    except:
        print("No unlabeled data available")
        unlabeled_loader = None
    
    # 1. Train Metric Learning Model
    print("Training Metric Learning Model...")
    metric_model = train_metric_learning_model(train_loader, val_loader)
    
    # 2. Train Self-Supervised Model
    print("Training Self-Supervised Model...")
    simclr_transform = SimCLRDataTransform(None)
    ss_train_dataset = SelfSupervisedDataset(train_dataset, transform=simclr_transform)
    ss_val_dataset = SelfSupervisedDataset(val_dataset, transform=simclr_transform)
    
    ss_train_loader = DataLoader(ss_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ss_val_loader = DataLoader(ss_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    simclr_model = train_simclr_model(ss_train_loader, ss_val_loader)
    
    # 3. Train Semi-Supervised Model
    print("Training Semi-Supervised Model...")
    num_classes = len(train_dataset.dataset.classes) if hasattr(train_dataset.dataset, 'classes') else 100
    
    if unlabeled_loader:
        semi_supervised_model = train_semi_supervised_model(train_loader, unlabeled_loader, val_loader, num_classes)
    else:
        print("Skipping semi-supervised training due to lack of unlabeled data")
        semi_supervised_model = None
    
    # 4. Use Pre-trained CLIP Model
    print("Extracting features with CLIP model...")
    clip_train_features, train_labels, train_ids = extract_clip_features(train_loader)
    clip_val_features, val_labels, val_ids = extract_clip_features(val_loader)
    clip_test_features, _, test_ids = extract_clip_features(test_loader)
    
    # 5. Train Swin Transformer Model
    print("Training Swin Transformer Model...")
    swin_model = SwinTransformerModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(swin_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        swin_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            logits, _ = swin_model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Validation
        swin_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                logits, _ = swin_model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(swin_model.state_dict(), 'best_swin_model.pth')
    
    # 6. Feature Extraction from All Models
    print("Extracting features from all models...")
    
    # Extract features from metric learning model
    metric_model.load_state_dict(torch.load('best_metric_model.pth'))
    metric_model.eval()
    
    metric_train_features = []
    metric_test_features = []
    
    with torch.no_grad():
        for images, _, _ in tqdm(train_loader):
            images = images.to(DEVICE)
            features = metric_model(images)
            metric_train_features.append(features.cpu().numpy())
        
        for images, _ in tqdm(test_loader):
            images = images.to(DEVICE)
            features = metric_model(images)
            metric_test_features.append(features.cpu().numpy())
    
    metric_train_features = np.vstack(metric_train_features)
    metric_test_features = np.vstack(metric_test_features)
    
    # Extract features from SimCLR model
    simclr_model.load_state_dict(torch.load('best_simclr_model.pth'))
    simclr_model.eval()
    
    simclr_train_features = []
    simclr_test_features = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            if isinstance(batch[0], tuple):  # Handle self-supervised data format
                (images1, _), labels, _ = batch
            else:
                images1, labels, _ = batch
            
            images1 = images1.to(DEVICE)
            features = simclr_model(images1)
            simclr_train_features.append(features.cpu().numpy())
        
        for images, _ in tqdm(test_loader):
            images = images.to(DEVICE)
            features = simclr_model(images)
            simclr_test_features.append(features.cpu().numpy())
    
    simclr_train_features = np.vstack(simclr_train_features)
    simclr_test_features = np.vstack(simclr_test_features)
    
    # Extract features from semi-supervised model if available
    if semi_supervised_model:
        semi_supervised_model.load_state_dict(torch.load('best_semi_supervised_model.pth'))
        semi_supervised_model.eval()
        
        semi_train_features = []
        semi_test_features = []
        
        with torch.no_grad():
            for images, _, _ in tqdm(train_loader):
                images = images.to(DEVICE)
                features = semi_supervised_model(images, return_features=True)
                semi_train_features.append(features.cpu().numpy())
            
            for images, _ in tqdm(test_loader):
                images = images.to(DEVICE)
                features = semi_supervised_model(images, return_features=True)
                semi_test_features.append(features.cpu().numpy())
        
        semi_train_features = np.vstack(semi_train_features)
        semi_test_features = np.vstack(semi_test_features)
    else:
        semi_train_features = None
        semi_test_features = None
    
    # Extract features from Swin model
    swin_model.load_state_dict(torch.load('best_swin_model.pth'))
    swin_model.eval()
    
    swin_train_features = []
    swin_test_features = []
    
    with torch.no_grad():
        for images, _, _ in tqdm(train_loader):
            images = images.to(DEVICE)
            _, features = swin_model(images)
            swin_train_features.append(features.cpu().numpy())
        
        for images, _ in tqdm(test_loader):
            images = images.to(DEVICE)
            _, features = swin_model(images)
            swin_test_features.append(features.cpu().numpy())
    
    swin_train_features = np.vstack(swin_train_features)
    swin_test_features = np.vstack(swin_test_features)
    
    # 7. Evaluate Models on Validation Set
    print("Evaluating models on validation set...")
    
    # Evaluate metric learning model
    metric_val_features = []
    with torch.no_grad():
        for images, _, _ in tqdm(val_loader):
            images = images.to(DEVICE)
            features = metric_model(images)
            metric_val_features.append(features.cpu().numpy())
    
    metric_val_features = np.vstack(metric_val_features)
    metric_predictions = retrieve_images(metric_val_features, metric_train_features, train_labels)
    metric_map5 = calculate_map5(metric_predictions, val_labels)
    print(f"Metric Learning Model MAP@5: {metric_map5:.4f}")
    
    # Evaluate SimCLR model
    simclr_val_features = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if isinstance(batch[0], tuple):
                (images1, _), labels, _ = batch
            else:
                images1, labels, _ = batch
            
            images1 = images1.to(DEVICE)
            features = simclr_model(images1)
            simclr_val_features.append(features.cpu().numpy())
    
    simclr_val_features = np.vstack(simclr_val_features)
    simclr_predictions = retrieve_images(simclr_val_features, simclr_train_features, train_labels)
    simclr_map5 = calculate_map5(simclr_predictions, val_labels)
    print(f"SimCLR Model MAP@5: {simclr_map5:.4f}")
    
    # Evaluate semi-supervised model if available
    if semi_supervised_model:
        semi_val_features = []
        with torch.no_grad():
            for images, _, _ in tqdm(val_loader):
                images = images.to(DEVICE)
                features = semi_supervised_model(images, return_features=True)
                semi_val_features.append(features.cpu().numpy())
        
        semi_val_features = np.vstack(semi_val_features)
        semi_predictions = retrieve_images(semi_val_features, semi_train_features, train_labels)
        semi_map5 = calculate_map5(semi_predictions, val_labels)
        print(f"Semi-Supervised Model MAP@5: {semi_map5:.4f}")
    else:
        semi_map5 = 0.0
    
    # Evaluate Swin model
    swin_val_features = []
    with torch.no_grad():
        for images, _, _ in tqdm(val_loader):
            images = images.to(DEVICE)
            _, features = swin_model(images)
            swin_val_features.append(features.cpu().numpy())
    
    swin_val_features = np.vstack(swin_val_features)
    swin_predictions = retrieve_images(swin_val_features, swin_train_features, train_labels)
    swin_map5 = calculate_map5(swin_predictions, val_labels)
    print(f"Swin Transformer Model MAP@5: {swin_map5:.4f}")
    
    # Evaluate CLIP model
    clip_predictions = retrieve_images(clip_val_features, clip_train_features, train_labels)
    clip_map5 = calculate_map5(clip_predictions, val_labels)
    print(f"CLIP Model MAP@5: {clip_map5:.4f}")
    
    # 8. Create Feature Ensemble
    print("Creating feature ensemble...")
    
    # Normalize features for fair contribution
    def normalize_features(features):
        return features / np.linalg.norm(features, axis=1, keepdims=True)
    
    metric_train_features_norm = normalize_features(metric_train_features)
    metric_val_features_norm = normalize_features(metric_val_features)
    metric_test_features_norm = normalize_features(metric_test_features)
    
    simclr_train_features_norm = normalize_features(simclr_train_features)
    simclr_val_features_norm = normalize_features(simclr_val_features)
    simclr_test_features_norm = normalize_features(simclr_test_features)
    
    swin_train_features_norm = normalize_features(swin_train_features)
    swin_val_features_norm = normalize_features(swin_val_features)
    swin_test_features_norm = normalize_features(swin_test_features)
    
    clip_train_features_norm = normalize_features(clip_train_features)
    clip_val_features_norm = normalize_features(clip_val_features)
    clip_test_features_norm = normalize_features(clip_test_features)
    
    if semi_supervised_model:
        semi_train_features_norm = normalize_features(semi_train_features)
        semi_val_features_norm = normalize_features(semi_val_features)
        semi_test_features_norm = normalize_features(semi_test_features)
    
    # Try different feature combinations with different weights
    best_map5 = 0.0
    best_weights = None
    best_predictions = None
    
    # Define a range of weights to try
    weight_ranges = np.linspace(0.1, 1.0, 10)
    
    for w1 in weight_ranges:
        for w2 in weight_ranges:
            for w3 in weight_ranges:
                for w4 in weight_ranges:
                    if semi_supervised_model:
                        for w5 in weight_ranges:
                            weights = np.array([w1, w2, w3, w4, w5])
                            weights = weights / weights.sum()  # Normalize to sum to 1
                            
                            # Combine features
                            ensemble_train_features = (
                                weights[0] * metric_train_features_norm +
                                weights[1] * simclr_train_features_norm +
                                weights[2] * semi_train_features_norm +
                                weights[3] * swin_train_features_norm +
                                weights[4] * clip_train_features_norm
                            )
                            
                            ensemble_val_features = (
                                weights[0] * metric_val_features_norm +
                                weights[1] * simclr_val_features_norm +
                                weights[2] * semi_val_features_norm +
                                weights[3] * swin_val_features_norm +
                                weights[4] * clip_val_features_norm
                            )
                    else:
                        weights = np.array([w1, w2, w3, w4])
                        weights = weights / weights.sum()  # Normalize to sum to 1
                        
                        # Combine features
                        ensemble_train_features = (
                            weights[0] * metric_train_features_norm +
                            weights[1] * simclr_train_features_norm +
                            weights[2] * swin_train_features_norm +
                            weights[3] * clip_train_features_norm
                        )
                        
                        ensemble_val_features = (
                            weights[0] * metric_val_features_norm +
                            weights[1] * simclr_val_features_norm +
                            weights[2] * swin_val_features_norm +
                            weights[3] * clip_val_features_norm
                        )
                    
                    # Evaluate ensemble
                    ensemble_predictions = retrieve_images(ensemble_val_features, ensemble_train_features, train_labels)
                    ensemble_map5 = calculate_map5(ensemble_predictions, val_labels)
                    
                    if ensemble_map5 > best_map5:
                        best_map5 = ensemble_map5
                        best_weights = weights
                        best_predictions = ensemble_predictions
    
    print(f"Best Ensemble MAP@5: {best_map5:.4f}")
    print(f"Best Weights: {best_weights}")
    
    # 9. Make Predictions on Test Set
    print("Making predictions on test set...")
    
    # Create final ensemble features for test set
    if semi_supervised_model:
        ensemble_test_features = (
            best_weights[0] * metric_test_features_norm +
            best_weights[1] * simclr_test_features_norm +
            best_weights[2] * semi_test_features_norm +
            best_weights[3] * swin_test_features_norm +
            best_weights[4] * clip_test_features_norm
        )
    else:
        ensemble_test_features = (
            best_weights[0] * metric_test_features_norm +
            best_weights[1] * simclr_test_features_norm +
            best_weights[2] * swin_test_features_norm +
            best_weights[3] * clip_test_features_norm
        )
    
    # Get predictions
    final_predictions = retrieve_images(ensemble_test_features, ensemble_train_features, train_labels)
    
    # Save submission
    save_submission(final_predictions, test_ids, 'submission.csv')
    print("Submission saved to submission.csv")
    
    # 10. Visualize Results
    print("Visualizing results...")
    
    # Plot MAP@5 scores
    model_names = ['Metric Learning', 'SimCLR', 'Swin Transformer', 'CLIP']
    map5_scores = [metric_map5, simclr_map5, swin_map5, clip_map5]
    
    if semi_supervised_model:
        model_names.append('Semi-Supervised')
        map5_scores.append(semi_map5)
    
    model_names.append('Ensemble')
    map5_scores.append(best_map5)
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, map5_scores)
    plt.xlabel('Models')
    plt.ylabel('MAP@5')
    plt.title('MAP@5 Scores for Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('map5_scores.png')
    
    # Plot feature space visualization (t-SNE)
    from sklearn.manifold import TSNE
    
    # Sample a subset of data for visualization
    n_samples = 1000
    indices = np.random.choice(len(train_labels), n_samples, replace=False)
    
    ensemble_train_features_subset = ensemble_train_features[indices]
    train_labels_subset = train_labels[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    ensemble_train_2d = tsne.fit_transform(ensemble_train_features_subset)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    
    # Get unique classes
    unique_classes = np.unique(train_labels_subset)
    
    # Plot each class with a different color
    for i, cls in enumerate(unique_classes):
        if i >= 10:  # Limit to 10 classes for clarity
            break
        
        mask = train_labels_subset == cls
        plt.scatter(ensemble_train_2d[mask, 0], ensemble_train_2d[mask, 1], label=f'Class {cls}', alpha=0.7)
    
    plt.legend()
    plt.title('t-SNE Visualization of Ensemble Features')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    
    print("Visualization saved to map5_scores.png and tsne_visualization.png")

if __name__ == "__main__":
    main()