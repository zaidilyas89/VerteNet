


from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

img = Image.open(r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\GuideNetv1_Outputs_Model\0.png').convert("RGB")
# plt.imshow(img)
img_1 = np.array(img)

img_1 = np.transpose(img_1,(2,0,1))
img_1 = np.expand_dims(img_1, axis=0)



# b_down_Hi, c_down_Hi, h_down_Hi, w_down_Hi= img_1.shape
# img_1 = rearrange(img_1[0,0,:,:], '(p1 h) (p2 w) ->(p1 p2) h w', p1=8, p2=8, h = 480//8, w = 640//8)
# img_1 = rearrange(img_1, 'b c (h head1) (w head2) -> b c h head1 w head2', head1 = 8, head2 = 8)
# img_1 = rearrange(img_1, 'b c h head1 w head2 -> b c (head1 head2) (h w)')
# img_1 = rearrange(img_1, 'b c head1_head2 h_w -> b c h_w head1_head2')

# img_1 = rearrange(img_1, 'b c h head1 w head2 -> b (head1 head2) c h w')


# a = img_1[0,0,:].transpose(1,2,0)
# plt.imshow(Image.fromarray(a))


# img_1 = rearrange(img_1, 'b c (h p1) (w p2) -> b (h*w) c p1 p2', p1=8, p2=8)


# def get_patches(image, patch_size, stride):
#     h, w = image.shape[:2]
#     patches = rearrange(image[:h // patch_size[0] * patch_size[0], :w // patch_size[1] * patch_size[1]],
#                        '(h_patch h) (w_patch w) -> (h_patch w_patch) h w',
#                        h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1])
#     return patches

# # def get_patches(image, patch_size, stride):
# #     h, w, c = image.shape
# #     patches = rearrange(image[:h // patch_size[0] * patch_size[0], :w // patch_size[1] * patch_size[1]],
# #                        '(h_patch h) (w_patch w) c -> (h_patch w_patch) h w c',
# #                        h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1], c=c)
# #     return patches


# def reconstruct_image(patches, image_shape, patch_size, stride):
#     h, w, c = image_shape
#     h_patches = (h - patch_size[0]) // stride + 1
#     w_patches = (w - patch_size[1]) // stride + 1
#     image = np.zeros((h, w, c), dtype=patches.dtype)
#     index = 0
#     for i in range(h_patches):
#         for j in range(w_patches):
#             image[i*stride:i*stride+patch_size[0], j*stride:j*stride+patch_size[1], :] = patches[index]
#             index += 1
#     return image

# im = get_patches(img_1[0,:,:], (8,8), 4)


# plt.imshow(Image.fromarray(im[36,:,:]))

########################################################
# import numpy as np
# from einops import rearrange

# from einops import rearrange

patch_size = 10

def get_patches(image, patch_size):
    image = image.transpose(0,2,3,1)
    b, h, w, c = image.shape
    patches = rearrange(image,
                       'b (h_patch h) (w_patch w) c -> b (h_patch w_patch) h w c',
                       h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1])
    return patches






def reconstruct_image(patches, image_shape, patch_size):
    b, h, w, c = image_shape
    image = rearrange(patches,
                       'b (h_patch w_patch) h w c -> b (h_patch h) (w_patch w) c',
                       h_patch=patch_size[0], w_patch=patch_size[1], h=h // patch_size[0], w=w // patch_size[1])
    return image

patches = get_patches(img_1, (patch_size,patch_size))

reconstructed_image = reconstruct_image(patches, (1,480,640,3), (patch_size,patch_size))

plt.figure(1)
plt.imshow(patches[0,43,:,:,:])

plt.figure(2)
plt.imshow(patches[0,44,:,:,:])

plt.figure(3)
plt.imshow(reconstructed_image[0,:,:,:])

plt.figure(4)
plt.imshow(img_1.transpose(0,2,3,1)[0,:,:,:])