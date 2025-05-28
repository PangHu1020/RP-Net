import torch
import scipy.io
import h5py
import os
from scipy.io import loadmat
import numpy as np
import scipy.fftpack
import numpy as np
from args_parser import parse_args
from torchvision import transforms
import cv2
from PIL import Image

# args = parse_args()


# class GRM:
#     def __init__(self, num=220,v=8,numPixels=128,number=64,Ndiscr_mon=6, path=args.simA):
#         self.num = num
#         self.v = v
#         self.numPixels = numPixels
#         self.number = number
#         self.Ndiscr_mon = Ndiscr_mon
#         filename = f'simA_{self.number}_{self.numPixels}.mat'
#         file_path = os.path.join(path, filename)
#         if os.path.exists(file_path):
#             with h5py.File(file_path, 'r') as f:
#                 self.simA = torch.from_numpy(f['simA'][:]).to(torch.float32)
#             # simA = data['simA']
#         else:
#             raise FileNotFoundError(f"File {file_path} does not exist.")
#         # self.simA = data['simA']
#         self.subblocksperaxis = torch.tensor([1, 1], dtype=torch.float32)

#     def __call__(self, img):
#         # x = torch.from_numpy(img).to(torch.float64)
#         x = torch.from_numpy(np.array(img)).to(torch.float32)
#         if x.size(0) != 3:
#             x = torch.permute(x, (2,0,1))

#         xr = x[0]
#         xg = x[1]
#         xb = x[2]

#         yr = self.simA @ (xr.reshape(-1,1))
#         yg = self.simA @ (xg.reshape(-1,1))
#         yb = self.simA @ (xb.reshape(-1,1))

#         yr = yr/torch.max(yr)
#         yg = yg/torch.max(yg)
#         yb = yb/torch.max(yb)

#         yr = yr.reshape(self.numPixels,self.numPixels)
#         yg = yg.reshape(self.numPixels,self.numPixels)
#         yb = yb.reshape(self.numPixels,self.numPixels)

#         yr = yr.unsqueeze(0)
#         yg = yg.unsqueeze(0)
#         yb = yb.unsqueeze(0)

#         y = torch.cat((yr,yg,yb),dim=0)

#         sr = self.v*1000/(self.Ndiscr_mon**2)*torch.prod(self.subblocksperaxis.clone().detach())
#         sg = self.v*1000/(self.Ndiscr_mon**2)*torch.prod(self.subblocksperaxis.clone().detach())
#         sb = self.v*1000/(self.Ndiscr_mon**2)*torch.prod(self.subblocksperaxis.clone().detach())


#         f = self.reconstruct_tv_it_cbg(
#             A=self.simA,
#             cscale=[sr,sg,sb],
#             measurement=y,
#             reg=[self.num, self.num, self.num],
#             NumBlocks=[self.number,self.number],
#             tik_reg=[1.0,1.0,1.0]
#         )

#         fr = f[0]
#         fg = f[1]
#         fb = f[2]

#         fr = fr/torch.max(fr)
#         fg = fg/torch.max(fg)
#         fb = fb/torch.max(fb)

#         fr = fr.unsqueeze(0)
#         fg = fg.unsqueeze(0)
#         fb = fb.unsqueeze(0)

#         f_tensor = torch.cat(tensors=(fr,fg,fb),dim=0)
    
#         return f_tensor

#     def dctmtx(self, n):
#         """
#         Generates a DCT (Discrete Cosine Transform) matrix of size n x n.
#         """
#         dct_matrix = np.zeros((n, n))
#         for k in range(n):
#             for i in range(n):
#                 if k == 0:
#                     alpha = np.sqrt(1 / n)
#                 else:
#                     alpha = np.sqrt(2 / n)
#                 dct_matrix[k, i] = alpha * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
#         return torch.from_numpy(dct_matrix).float()   
    
#     def reconstruct_tv_it_cbg(self, A, cscale, measurement, reg, NumBlocks, tik_reg):
#         """
#         Reconstructs the scene given a measurement image and the A matrix using TV regularization.
        
#         Parameters:
#             A           : Light transport matrix (PyTorch tensor)
#             cscale      : Color scaling factors (list or tensor of length 3)
#             measurement : Measurement image (PyTorch tensor of shape [height, width, 3])
#             reg         : Regularization parameters for each color channel (list or tensor of length 3)
#             NumBlocks   : Number of blocks in the reconstruction grid (list or tensor of length 2)
#             tik_reg     : Tikhonov regularization parameters for each color channel (list or tensor of length 3)
        
#         Returns:
#             reconstruction : Reconstructed image (PyTorch tensor of shape [height, width, 3])
#         """
#         # Difference matrix
#         n = A.shape[0]
#         Diff = torch.eye(n, dtype=A.dtype) - torch.diag(torch.ones(n - 1, dtype=A.dtype), diagonal=1)

#         # Reshape the measurement image
#         mr = Diff @ measurement[0].flatten()
#         mg = Diff @ measurement[1].flatten()
#         mb = Diff @ measurement[2].flatten()

#         dA = Diff @ A
#         AT = dA.T
#         ATA = AT @ dA


#         # Initial estimates
#         if any(tik_reg):
#             # Tikhonov regularization with 1/f DCT prior
#             weight1 = torch.linspace(0.1, 1, steps=NumBlocks[1]) ** 0.35
#             weight2 = torch.linspace(0.1, 1, steps=NumBlocks[0]) ** 0.35
#             weight1 = weight1 / weight1.max()
#             weight2 = weight2 / weight2.max()
#             D = torch.kron(torch.diag(weight1) @ self.dctmtx(NumBlocks[1]),
#                         torch.diag(weight2) @ self.dctmtx(NumBlocks[0]))
#             DtD = D.T @ D
#         else:
#             DtD = torch.zeros_like(ATA)

#         # Function to reshape vector to image
#         def rs(input_vec):
#             return input_vec.reshape(NumBlocks[0], NumBlocks[1])

#         # Solve for initial estimates
#         testr = rs(torch.linalg.solve(cscale[0]**2 * ATA + tik_reg[0]**4 * DtD, cscale[0]*AT @ mr))
#         testg = rs(torch.linalg.solve(cscale[1]**2 * ATA + tik_reg[1]**4 * DtD, cscale[1]*AT @ mg))
#         testb = rs(torch.linalg.solve(cscale[2]**2 * ATA + tik_reg[2]**4 * DtD, cscale[2]*AT @ mb))

#         reconstruction = torch.stack([testr, testg, testb], dim=0)

#         return reconstruction





class DiffractionNoise_priori:
    def __init__(self, radius=30, noise_scale=0.1):
        self.radius = radius
        self.noise_scale = noise_scale

    def __call__(self, image):
        # Convert PIL Image to numpy array (assuming input image is PIL)
        image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Generate noise
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=image_np.shape)

        # Fourier Transform to create diffraction effect
        f = np.fft.fft2(noise)
        fshift = np.fft.fftshift(f)

        # Create mask to simulate diffraction grating
        rows, cols = image_np.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), self.radius, 1, -1)

        if image_np.ndim == 3 and image_np.shape[2] == 3:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Apply mask to the noise in the frequency domain
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        diffraction_noise = np.fft.ifft2(f_ishift)
        diffraction_noise = np.abs(diffraction_noise)

        # Add the diffraction noise to the original image
        result = image_np + diffraction_noise
        result = np.clip(result, 0, 1)  # Ensure values are in [0, 1]

        # Convert numpy array back to PIL Image
        result = (result * 255).astype(np.uint8)
        return transforms.functional.to_pil_image(result)
    


class DiffuseNoise_priori:
    def __init__(self, noise_scale=0.05, kernel_size=15):
        """
        :param noise_scale: 控制漫反射噪声的强度，值越大噪声越明显
        :param kernel_size: 控制噪声的模糊程度，越大则模拟越多的漫反射效果
        """
        self.noise_scale = noise_scale
        self.kernel_size = kernel_size

    def __call__(self, image):
        # Convert PIL Image to numpy array (assuming input image is PIL)
        image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Generate random noise
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=image_np.shape).astype(np.float32)

        # Apply Gaussian blur to the noise to simulate diffuse reflection
        noise = cv2.GaussianBlur(noise, (self.kernel_size, self.kernel_size), 0)

        # Add the diffuse noise to the original image
        noisy_image = image_np + noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure values are in [0, 1]

        # Convert numpy array back to PIL Image
        noisy_image = (noisy_image * 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

