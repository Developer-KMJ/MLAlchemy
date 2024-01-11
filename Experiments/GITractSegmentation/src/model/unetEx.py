import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

          # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.down_block_1 = self.double_conv_block(3, 64) # 572x572x3 -> 570x570x64 -> 568x568x64 -> 284x284x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 284x284x64
        self.down_block_2 = self.double_conv_block(64, 128) # 284x284x64 -> 282x282x128 -> 280x280x128 -> 140x140x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        
        # input: 140x140x128
        self.down_block_3 = self.double_conv_block(128, 256) # 140x140x128 -> 138x138x256 -> 136x136x256 -> 68x68x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 68x68x256
        self.down_block_4 = self.double_conv_block(256, 512) # 68x68x256 -> 66x66x512 -> 64x64x512 -> 32x32x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = self.double_conv_block(512, 1024) # 32x32x512 -> 30x30x1024 -> 28x28x1024
        

        # Decoder
        self.up_conv_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_block_4 = self.double_conv_block(1024, 512)
       
        self.up_conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_block_3 = self.double_conv_block(512, 256)

        self.up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block_2 = self.double_conv_block(256, 128)
        
        self.up_conv_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block_1 = self.double_conv_block(128, 64)
       
        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1, padding='same')

    def double_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(True),
        )
            
    def crop_to_match_target(self, residual, target):

        e1 = (int)((residual.shape[2] - target.shape[2])/2)
        e2 = (int)(residual.shape[2] - ((residual.shape[2] - target.shape[2]) - e1))

        e3 = (int)((residual.shape[3] - target.shape[3])/2)
        e4 = (int)(residual.shape[3] - ((residual.shape[3] - target.shape[3]) - e3))

        cropped_residual = residual[:, :, e1:e2, e3:e4]
        return cropped_residual


    def forward(self, x):
        # Encoder

        # Block 1
        x = self.down_block_1(x)
        b1_residual = x
        x = self.pool1(x)
        
        x = self.down_block_2(x)
        b2_residual = x
        x = self.pool2(x)

        x = self.down_block_3(x) 
        b3_residual = x
        x = self.pool3(x)

        x = self.down_block_4(x)
        b4_residual = x
        x = self.pool4(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up_conv_4(x)
        cropped_b4_residual = self.crop_to_match_target(b4_residual, x)
        x = torch.cat([x, cropped_b4_residual], dim=1)
        x = self.up_block_4(x)

        x = self.up_conv_3(x)
        cropped_b3_residual = self.crop_to_match_target(b3_residual, x)
        x = torch.cat([x, cropped_b3_residual], dim=1)
        x = self.up_block_3(x)

        x = self.up_conv_2(x)
        cropped_b2_residual = self.crop_to_match_target(b2_residual, x)
        x = torch.cat([x, cropped_b2_residual], dim=1)
        x = self.up_block_2(x)

        x = self.up_conv_1(x)
        cropped_b1_residual = self.crop_to_match_target(b1_residual, x)
        x = torch.cat([x, cropped_b1_residual], dim=1)
        x = self.up_block_1(x)
    
        # Output layer
        out = self.outconv(x)

        return out





def main():


    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(num_classes=10)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)

    print(outputs.shape)

if __name__ == "__main__":
    main()
    

