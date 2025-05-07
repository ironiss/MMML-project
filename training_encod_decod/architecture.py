import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownsampleBlock(nn.Module):
    def __init__(self, scale_factor):
        super(DownsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
    
class MAWarpBlock(nn.Module):
    def __init__(self, feature_channels):
        super(MAWarpBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_channels + 4, feature_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels)
        
    def forward(self, features, motion_field):
        x = torch.cat([features, motion_field], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        
        flat_z = z.view(-1, self.embedding_dim)
        
        distances = torch.sum(flat_z**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_z, self.embedding.weight.t())
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        quantized_st = z + (quantized - z).detach()
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()
        
        vq_loss = F.mse_loss(quantized.detach(), z.permute(0, 3, 1, 2).contiguous())
        commitment_loss = F.mse_loss(quantized, z.permute(0, 3, 1, 2).detach().contiguous())
        
        return quantized_st, vq_loss, commitment_loss, encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super(Encoder, self).__init__()
        self.conv_init = ConvBlock(in_channels, hidden_dim)
        
        self.res_blk1 = ResBlock(hidden_dim)
        self.conv1 = ConvBlock(hidden_dim, hidden_dim)
        self.down1 = DownsampleBlock(0.5) 
        
        
        self.res_blk2 = ResBlock(hidden_dim)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim * 2)
        self.down2 = DownsampleBlock(0.5) 
        
        self.res_blk3 = ResBlock(hidden_dim * 2)
        self.conv3 = ConvBlock(hidden_dim * 2, hidden_dim * 4)
        self.down3 = DownsampleBlock(0.5)  
        
        self.res_blk4 = ResBlock(hidden_dim * 4)
        self.conv4 = ConvBlock(hidden_dim * 4, hidden_dim * 8)
        self.down4 = DownsampleBlock(0.5)  
        
        self.res_blk5 = ResBlock(hidden_dim * 8)
        self.conv5 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        self.down5 = DownsampleBlock(0.5) 
        
        self.res_blk6 = ResBlock(hidden_dim * 8)
        self.conv6 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        
        self.vq_layer = VectorQuantizer(512, hidden_dim * 8)
        
    def forward(self, x):
        x = self.conv_init(x)
        
        x = self.res_blk1(x)
        x = self.conv1(x)
        x = self.down1(x)
        
        x = self.res_blk2(x)
        x = self.conv2(x)
        x = self.down2(x)
        
        x = self.res_blk3(x)
        x = self.conv3(x)
        x = self.down3(x)
        
        x = self.res_blk4(x)
        x = self.conv4(x)
        x = self.down4(x)
        
        x = self.res_blk5(x)
        x = self.conv5(x)
        x = self.down5(x)
        
        x = self.res_blk6(x)
        enc_features = self.conv6(x)
        
        quantized, vq_loss, commitment_loss, encoding_indices = self.vq_layer(enc_features)
        
        return enc_features, quantized, vq_loss, commitment_loss, encoding_indices
    

class EncoderWithFeatures(nn.Module):
    def __init__(self, encoder):
        super(EncoderWithFeatures, self).__init__()
        self.encoder = encoder
        
    def forward(self, x):
        features = {}
        
        x = self.encoder.conv_init(x)
        
        x = self.encoder.res_blk1(x)
        x = self.encoder.conv1(x)
        x = self.encoder.down1(x)
        
        x = self.encoder.res_blk2(x)
        x = self.encoder.conv2(x)
        x = self.encoder.down2(x)
        
        features['block2'] = x.clone()
        
        x = self.encoder.res_blk3(x)
        x = self.encoder.conv3(x)
        x = self.encoder.down3(x)
        
        features['block3'] = x.clone()
        
        x = self.encoder.res_blk4(x)
        x = self.encoder.conv4(x)
        x = self.encoder.down4(x)
        
        features['block4'] = x.clone()
        
        x = self.encoder.res_blk5(x)
        x = self.encoder.conv5(x)
        x = self.encoder.down5(x)
        
        x = self.encoder.res_blk6(x)
        enc_features = self.encoder.conv6(x)
        
        quantized, vq_loss, commitment_loss, encoding_indices = self.encoder.vq_layer(enc_features)
        
        return enc_features, quantized, vq_loss, commitment_loss, encoding_indices, features


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=64):
        super(Decoder, self).__init__()
        self.initial_conv = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        
        self.fusion_minus1 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1)
        self.fusion_plus1 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=1)
        
        self.up1 = UpsampleBlock()
        self.ma_warp1 = MAWarpBlock(hidden_dim * 8)
        self.conv1 = ConvBlock(hidden_dim * 8, hidden_dim * 8)
        self.res_blk1 = ResBlock(hidden_dim * 8)
        
        self.up2 = UpsampleBlock()
        self.ma_warp2 = MAWarpBlock(hidden_dim * 8)
        self.conv2 = ConvBlock(hidden_dim * 8, hidden_dim * 4)
        self.res_blk2 = ResBlock(hidden_dim * 4)
        
        self.up3 = UpsampleBlock()
        self.ma_warp3 = MAWarpBlock(hidden_dim * 4)
        
        self.fusion_conv = nn.Conv2d(hidden_dim * 4 + hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(hidden_dim * 4)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        self.conv3 = ConvBlock(hidden_dim * 4, hidden_dim * 2)
        self.res_blk3 = ResBlock(hidden_dim * 2)
        
        self.up4 = UpsampleBlock()
        self.ma_warp4 = MAWarpBlock(hidden_dim * 2)
        self.conv4 = ConvBlock(hidden_dim * 2, hidden_dim)
        self.res_blk4 = ResBlock(hidden_dim)
        
        self.up5 = UpsampleBlock()
        self.ma_warp5 = MAWarpBlock(hidden_dim)
        self.conv5 = ConvBlock(hidden_dim, hidden_dim)
        self.res_blk5 = ResBlock(hidden_dim)
        
        self.up6 = UpsampleBlock()
        self.ma_warp6 = MAWarpBlock(hidden_dim)
        self.conv6 = ConvBlock(hidden_dim, hidden_dim)
        self.res_blk6 = ResBlock(hidden_dim)
        
        self.output = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, z, motion_field, phi_minus1=None, phi_plus1=None):
        z = self.initial_conv(z)
        
        m_small = F.interpolate(motion_field, size=z.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.up1(z)
        x = self.ma_warp1(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv1(x)
        x = self.res_blk1(x)
        
        x = self.up2(x)
        x = self.ma_warp2(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv2(x)
        x = self.res_blk2(x)
        
        x = self.up3(x)
        x = self.ma_warp3(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        
        if phi_minus1 is not None and phi_plus1 is not None:
            phi_minus1_processed = self.fusion_minus1(phi_minus1)
            phi_plus1_processed = self.fusion_plus1(phi_plus1)
            
            phi_minus1_resized = F.interpolate(phi_minus1_processed, size=x.shape[2:], mode='bilinear', align_corners=False) 
            phi_plus1_resized = F.interpolate(phi_plus1_processed, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            fused_features = torch.cat([x, phi_minus1_resized, phi_plus1_resized], dim=1)
            
            x = self.fusion_conv(fused_features)
            x = self.fusion_relu(self.fusion_bn(x))
        
        x = self.conv3(x)
        x = self.res_blk3(x)
        
        x = self.up4(x)
        x = self.ma_warp4(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv4(x)
        x = self.res_blk4(x)
        
        x = self.up5(x)
        x = self.ma_warp5(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv5(x)
        x = self.res_blk5(x)
        
        x = self.up6(x)
        x = self.ma_warp6(x, F.interpolate(m_small, size=x.shape[2:], mode='bilinear', align_corners=False))
        x = self.conv6(x)
        x = self.res_blk6(x)
        
        x = self.tanh(self.output(x))
        
        return x
    
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)