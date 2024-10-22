class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, smoothing=0.1, lambda_entropy=0.02):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing
        self.lambda_entropy = lambda_entropy

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        # Calculate entropy of the predictions to add confidence regularization
        probs_img = F.softmax(logits, dim=1)
        entropy_img = -torch.sum(probs_img * torch.log(probs_img + 1e-8), dim=1).mean()

        probs_txt = F.softmax(logits.t(), dim=1)
        entropy_txt = -torch.sum(probs_txt * torch.log(probs_txt + 1e-8), dim=1).mean()

        # Combine the losses with confidence regularization
        entropy_penalty = (entropy_img + entropy_txt) / 2
        total_loss = (loss_img + loss_txt) / 2 - self.lambda_entropy * entropy_penalty

        return total_loss